import asyncio
import time

import pytest

from gradio.processing.memory import admission
from gradio.processing.pool import Pool


@pytest.fixture
def pool(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("""import os,time
import gradio as gr
app=gr.App()
@app.resource()
def model(): return []
@app.queue()
def predict(delay=0.1):
    values=model();values.append(1)
    time.sleep(delay)
    yield {"pid":os.getpid(),"count":len(values)}
if __name__=='__main__':app.launch()
""")
    pool = Pool(str(path), backlog_seconds=0.1, idle_seconds=1, cooldown_seconds=1)
    yield pool
    pool.close()


async def call(pool, delay=0.1):
    return [x async for x in pool.events_stream(0, (delay,), {})][-1]


@pytest.mark.asyncio
async def test_reuse_scale_and_idle(pool):
    first = [await call(pool, 0.4) for _ in range(3)]
    assert len({x["pid"] for x in first}) == 1
    assert [x["count"] for x in first] == [1, 2, 3]
    results = await asyncio.gather(*(call(pool, 0.4) for _ in range(16)))
    assert len({x["pid"] for x in results}) == 2
    assert all(w["resource_initializations"] == [1] for w in pool.status()["workers"])
    await asyncio.sleep(1.5)
    assert len(pool.status()["workers"]) == 1
    assert (await call(pool))["pid"] == first[0]["pid"]


@pytest.mark.asyncio
async def test_explicit_one(pool):
    pool.max_workers = 1
    results = await asyncio.gather(*(call(pool, 0.1) for _ in range(8)))
    assert len({x["pid"] for x in results}) == 1
    assert pool.status()["workers"][0]["resource_initializations"] == [1]


@pytest.mark.asyncio
async def test_cancel_and_recover(pool):
    await call(pool)
    task = asyncio.create_task(call(pool, 30))
    await asyncio.sleep(0.3)
    start = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - start < 5
    assert not pool.workers
    assert (await call(pool))["count"] == 1


def test_memory_budget():
    gb = 1024**3
    snap = {
        "kind": "cuda",
        "total": 142 * gb,
        "available": 100 * gb,
        "gpu_total": 80 * gb,
        "gpu_available": 50 * gb,
    }
    assert admission(snap, 4 * gb, 14 * gb)[0]
    assert not admission({**snap, "gpu_available": 16 * gb}, 4 * gb, 64 * gb)[0]
    assert not admission(
        {"kind": "unified", "total": 32 * gb, "available": 10 * gb}, 14 * gb, 0
    )[0]
    assert not admission({"kind": "cpu", "total": 32 * gb, "available": 20 * gb}, 0, 0)[
        0
    ]


@pytest.mark.asyncio
async def test_memory_denial_preserves_first_resource(pool):
    for _ in range(3):
        await call(pool, 0.2)
    original = pool.workers[0].process.pid
    original_sample = pool.memory.sample

    def constrained(pids):
        result = original_sample(pids)
        result["available"] = 1
        return result

    pool.memory.sample = constrained
    results = await asyncio.gather(*(call(pool, 0.2) for _ in range(12)))
    assert {x["pid"] for x in results} == {original}
    assert pool.workers[0].info["resource_initializations"] == [1]
    assert any("Insufficient" in e["reason"] for e in pool.events)


@pytest.mark.asyncio
async def test_high_concurrency_does_not_starve_stream_readers(pool):
    pool.max_workers = 1
    results = await asyncio.wait_for(
        asyncio.gather(*(call(pool, 0.01) for _ in range(64))), timeout=20
    )
    assert len(results) == 64
    assert len({x["pid"] for x in results}) == 1


@pytest.mark.asyncio
async def test_startup_exit_surfaces_error(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text("""import gradio as gr
app=gr.App()
@app.resource()
def model(): raise SystemExit(1)
@app.queue()
def predict(): return 1
""")
    pool = Pool(str(path))
    try:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(call(pool), timeout=10)
    finally:
        pool.close()


@pytest.mark.parametrize(
    "body",
    [
        "def predict(value): return value + 1",
        "async def predict(value): return value + 1",
        "async def predict(value):\n    yield value\n    yield value + 1",
    ],
)
@pytest.mark.asyncio
async def test_function_kinds(tmp_path, body):
    path = tmp_path / "kinds.py"
    path.write_text("import gradio as gr\napp=gr.App()\n@app.queue()\n" + body)
    pool = Pool(str(path), max_workers=1)
    try:
        result = [x async for x in pool.events_stream(0, (41,), {})]
        assert result[-1] == 42
    finally:
        pool.close()


@pytest.mark.asyncio
async def test_memory_pressure_drains_replica(pool):
    pool.idle_seconds = 60
    for _ in range(3):
        await call(pool, 0.2)
    await asyncio.gather(*(call(pool, 0.2) for _ in range(20)))
    assert len(pool.workers) == 2
    original_sample = pool.memory.sample

    def constrained(pids):
        result = original_sample(pids)
        result["available"] = 1
        return result

    pool.memory.sample = constrained
    for _ in range(20):
        if len(pool.workers) == 1:
            break
        await asyncio.sleep(0.1)
    assert len(pool.workers) == 1
    assert any("Memory reserve depleted" in e["reason"] for e in pool.events)
