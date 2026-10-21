"""Two-replica autoscaling PoC. Real jobs are never replayed for calibration."""

import asyncio
import atexit
import concurrent.futures
import functools
import inspect
import os
import secrets
import signal
import statistics
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener

from gradio.processing.memory import Memory, admission


class Worker:
    def __init__(self):
        self.process = None
        self.connection = None
        self.ready = False
        self.busy = False
        self.retiring = False
        self.peak_ram = 0
        self.peak_gpu = 0
        self.info = {}
        self.completed = 0

    def close(self):
        if self.process and self.process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.connection:
            self.connection.close()

    def start(self, path, environment):
        auth = secrets.token_bytes(32)
        with Listener(("127.0.0.1", 0), authkey=auth) as listener:
            listener._listener._socket.settimeout(1)
            env = dict(
                {**os.environ, **environment},
                GRADIO_PROCESS_WORKER="1",
                GRADIO_PROCESS_APP=path,
                GRADIO_PROCESS_PORT=str(listener.address[1]),
                GRADIO_PROCESS_AUTH=auth.hex(),
                TOKENIZERS_PARALLELISM="false",
            )
            self.process = subprocess.Popen(
                [sys.executable, "-m", "gradio.processing.worker"],
                env=env,
                start_new_session=True,
            )
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Worker exited during initialization")
                try:
                    self.connection = listener.accept()
                    break
                except TimeoutError:
                    continue
            if self.connection is None:
                raise TimeoutError("Worker did not connect")
        while time.monotonic() < deadline:
            if self.connection.poll(0.25):
                kind, payload = self.connection.recv()
                if kind != "ready":
                    raise RuntimeError(payload)
                self.info = payload
                self.ready = True
                return
            if self.process.poll() is not None:
                raise RuntimeError("Worker exited during model initialization")
        raise TimeoutError("Resource initialization exceeded 10 minutes")


class Pool:
    def __init__(
        self,
        path,
        max_workers=2,
        min_samples=3,
        backlog_seconds=0.5,
        idle_seconds=60,
        cooldown_seconds=60,
    ):
        self.path = os.path.abspath(path)
        self.environment = {}
        self.max_workers = max_workers
        self.min_samples = min_samples
        self.backlog_seconds = backlog_seconds
        self.idle_seconds = idle_seconds
        self.cooldown_seconds = cooldown_seconds
        self.memory = Memory()
        self.condition = threading.Condition(threading.RLock())
        self.workers = []
        self.pending = 0
        self.closed = False
        self.starting = False
        self.failure = None
        self.samples = []
        self.events = []
        self.last_activity = time.monotonic()
        self.backlog_since = None
        self.cooldown_until = 0
        self.trial = None
        self.snapshot = {}
        self._thread = threading.Thread(target=self.monitor, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def configure(self, environment, max_workers=2):
        """Demo-only reset between independent experiments; reject active work."""
        with self.condition:
            if self.starting or self.pending or any(w.busy for w in self.workers):
                raise RuntimeError("Cannot reconfigure an active pool")
            for worker in self.workers:
                worker.close()
            self.workers.clear()
            self.samples.clear()
            self.events.clear()
            self.environment = dict(environment)
            self.max_workers = max_workers
            self.failure = None
            self.trial = None
            self.cooldown_until = 0

    def record(self, reason, **extra):
        if self.events and self.events[-1]["reason"] == reason:
            return
        self.events.append({"time": time.time(), "reason": reason, **extra})
        self.events = self.events[-100:]

    def status(self):
        with self.condition:
            return {
                "policy_version": "poc-3",
                "pending": self.pending,
                "failure": self.failure,
                "starting": self.starting,
                "max_workers": self.max_workers,
                "memory": self.snapshot,
                "events": list(self.events),
                "workers": [
                    {
                        **w.info,
                        "ready": w.ready,
                        "busy": w.busy,
                        "retiring": w.retiring,
                        "completed": w.completed,
                        "peak_ram": w.peak_ram,
                        "peak_gpu": w.peak_gpu,
                    }
                    for w in self.workers
                ],
                "baseline_seconds": statistics.median(self.samples)
                if self.samples
                else None,
            }

    def spawn(self):
        # Caller holds condition. Only one worker may initialize at a time.
        self.starting = True
        worker = Worker()
        self.workers.append(worker)
        self.record("Initializing replica", workers=len(self.workers))

        def initialize():
            try:
                worker.start(self.path, self.environment)
                with self.condition:
                    if self.closed:
                        worker.close()
                        return
                    if len(self.workers) == 2:
                        self.trial = {
                            "start": None,
                            "durations": [],
                            "baseline": statistics.median(self.samples),
                            "continuous": False,
                        }
                    self.record("Replica ready", pid=worker.process.pid)
            except Exception as exc:  # noqa: BLE001 -- isolate arbitrary worker/provider failures
                worker.close()
                with self.condition:
                    if worker in self.workers:
                        self.workers.remove(worker)
                    self.record("Replica initialization failed", error=str(exc))
                    self.cooldown_until = time.monotonic() + self.cooldown_seconds
                    if not self.workers:
                        self.failure = str(exc) or repr(exc)
            finally:
                with self.condition:
                    self.starting = False
                    self.condition.notify_all()

        threading.Thread(target=initialize, daemon=True).start()

    def monitor(self):
        while not self.closed:
            try:
                with self.condition:
                    workers = list(self.workers)
                snap = self.memory.sample([w.process.pid for w in workers if w.process])
                with self.condition:
                    self.snapshot = snap
                    now = time.monotonic()
                    for w in workers:
                        if w.process:
                            usage = snap["processes"].get(w.process.pid, {})
                            w.peak_ram = max(w.peak_ram, usage.get("ram", 0))
                            w.peak_gpu = max(w.peak_gpu, usage.get("gpu", 0))
                    if self.pending and any(w.busy for w in workers):
                        self.backlog_since = self.backlog_since or now
                    else:
                        self.backlog_since = None
                    if (
                        len(workers) == 1
                        and self.max_workers > 1
                        and not self.starting
                        and len(self.samples) >= self.min_samples
                        and self.backlog_since
                        and now - self.backlog_since >= self.backlog_seconds
                        and now >= self.cooldown_until
                    ):
                        allowed, reason = admission(
                            snap, workers[0].peak_ram, workers[0].peak_gpu
                        )
                        self.record(reason)
                        if allowed:
                            self.spawn()
                    if (
                        len(workers) == 2
                        and not self.starting
                        and not workers[-1].retiring
                    ):
                        ram_pressure = snap["available"] < max(
                            1024**3, snap["total"] * 0.15
                        )
                        gpu_pressure = snap.get("gpu_available", float("inf")) < max(
                            1024**3, snap.get("gpu_total", 0) * 0.15
                        )
                        if ram_pressure or gpu_pressure:
                            self.retire(
                                workers[-1],
                                "Memory reserve depleted; draining extra replica",
                            )
                    if (
                        self.trial
                        and not self.trial["continuous"]
                        and self.pending
                        and all(w.busy and w.completed > 0 for w in workers)
                    ):
                        self.trial.update(start=now, durations=[], continuous=True)
                    if (
                        self.trial
                        and not self.pending
                        and not all(w.busy for w in workers)
                    ):
                        self.trial["continuous"] = False
                    if (
                        len(workers) == 2
                        and not self.starting
                        and now - self.last_activity > self.idle_seconds
                        and not self.pending
                        and not any(w.busy for w in workers)
                    ):
                        self.retire(workers[-1], "Idle replica retired")
            except Exception as exc:  # noqa: BLE001 -- isolate arbitrary worker/provider failures
                with self.condition:
                    self.record(
                        "Telemetry failed; expansion disabled for this sample",
                        error=str(exc),
                    )
            time.sleep(0.25)

    def retire(self, worker, reason):
        worker.retiring = True
        self.cooldown_until = time.monotonic() + self.cooldown_seconds
        self.trial = None
        self.record(reason)
        if not worker.busy:
            worker.close()
            if worker in self.workers:
                self.workers.remove(worker)

    def run(self, index, args, kwargs, cancelled):
        worker = None
        success = False
        with self.condition:
            self.pending += 1
            if (
                not self.workers
                and not self.starting
                and not self.failure
                and not self.closed
            ):
                self.spawn()
            try:
                while worker is None:
                    if cancelled.is_set() or self.closed:
                        return
                    if self.failure:
                        raise RuntimeError(self.failure)
                    worker = next(
                        (
                            w
                            for w in self.workers
                            if w.ready and not w.busy and not w.retiring
                        ),
                        None,
                    )
                    if worker:
                        worker.busy = True
                    else:
                        self.condition.wait(0.2)
            finally:
                self.pending -= 1
        try:
            worker.connection.send((index, args, kwargs))
            while not cancelled.is_set():
                if not worker.connection.poll(0.2):
                    if worker.process.poll() is not None:
                        raise RuntimeError("Worker exited; request was not retried")
                    continue
                kind, value = worker.connection.recv()
                if kind == "output":
                    yield value
                elif kind == "done":
                    success = True
                    with self.condition:
                        worker.completed += 1
                        worker.info["resource_initializations"] = value[
                            "resource_initializations"
                        ]
                        if (
                            len(self.workers) == 1
                            and not self.starting
                            and worker.completed > 1
                        ):
                            self.samples.append(value["seconds"])
                            self.samples = self.samples[-20:]
                        if self.trial and self.trial["start"] is not None:
                            self.trial["durations"].append(value["seconds"])
                            if (
                                len(self.trial["durations"]) >= 6
                                and self.trial["continuous"]
                            ):
                                throughput = len(self.trial["durations"]) / (
                                    time.monotonic() - self.trial["start"]
                                )
                                baseline = 1 / self.trial["baseline"]
                                latency = statistics.median(self.trial["durations"])
                                self.record(
                                    "Replica performance observed",
                                    throughput=throughput,
                                    baseline_throughput=baseline,
                                    median_seconds=latency,
                                )
                                if (
                                    throughput < baseline * 1.1
                                    or latency > self.trial["baseline"] * 2.5
                                ):
                                    self.retire(
                                        self.workers[-1],
                                        "Extra replica did not improve sustained throughput/latency",
                                    )
                                else:
                                    self.record(
                                        "Extra replica retained: throughput improved"
                                    )
                                    self.trial = None
                    return
                elif kind == "error":
                    # Treat arbitrary user failures conservatively; don't replay them.
                    raise RuntimeError(value)
        finally:
            with self.condition:
                worker.busy = False
                self.last_activity = time.monotonic()
                if not success or worker.retiring:
                    worker.close()
                    if worker in self.workers:
                        self.workers.remove(worker)
                    self.trial = None
                    if not self.workers and self.pending and not self.closed:
                        self.spawn()
                self.condition.notify_all()

    async def events_stream(self, index, args, kwargs):
        # One thread owns the entire generator. Scheduling each next() separately
        # can deadlock when waiting requests exhaust asyncio's executor threads.
        cancelled = threading.Event()
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=8)

        def deliver(message):
            future = asyncio.run_coroutine_threadsafe(queue.put(message), loop)
            while not cancelled.is_set():
                try:
                    future.result(timeout=0.2)
                    return True
                except concurrent.futures.TimeoutError:
                    pass
            future.cancel()
            return False

        def pump():
            iterator = self.run(index, args, kwargs, cancelled)
            try:
                for item in iterator:
                    if not deliver(("output", item)):
                        break
            except Exception as exc:  # noqa: BLE001 -- isolate arbitrary worker/provider failures
                deliver(("error", exc))
            finally:
                iterator.close()
                deliver(("done", None))

        task = asyncio.create_task(asyncio.to_thread(pump))
        try:
            while True:
                kind, item = await queue.get()
                if kind == "done":
                    return
                if kind == "error":
                    raise item
                yield item
        finally:
            cancelled.set()
            import anyio

            with anyio.CancelScope(shield=True):
                await asyncio.shield(task)

    def wrap(self, fn, index):
        if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def stream(*args, **kwargs):
                async for item in self.events_stream(index, args, kwargs):
                    yield item

            return stream

        @functools.wraps(fn)
        async def call(*args, **kwargs):
            result = None
            async for item in self.events_stream(index, args, kwargs):
                result = item
            return result

        return call

    def close(self):
        with self.condition:
            self.closed = True
            for worker in self.workers:
                worker.close()
            self.condition.notify_all()
