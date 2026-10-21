"""Exercise the actual HTTP queue and save raw evidence for every request."""

import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:7869")
parser.add_argument("--workload", default="python")
parser.add_argument("--requests", type=int, default=None)
parser.add_argument("--output", required=True)
args = parser.parse_args()
args.requests = args.requests or (256 if args.workload == "kokoro" else 24)
headers = {}
if os.environ.get("AUTOSCALE_CONTROL_TOKEN"):
    headers["x-control-token"] = os.environ["AUTOSCALE_CONTROL_TOKEN"]
if ".hf.space" in args.url:
    from huggingface_hub import get_token

    headers["Authorization"] = "Bearer " + get_token()


def call(_=None):
    start = time.monotonic()
    r = requests.post(
        args.url + "/gradio_api/call/predict",
        json={"data": ["Explain why the sky is blue in a few sentences."]},
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    event_id = r.json()["event_id"]
    outputs = []
    with requests.get(
        args.url + "/gradio_api/call/predict/" + event_id,
        headers=headers,
        stream=True,
        timeout=900,
    ) as stream:
        stream.raise_for_status()
        event = None
        for line in stream.iter_lines(decode_unicode=True):
            if line.startswith("event: "):
                event = line[7:]
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                if event == "error":
                    raise RuntimeError(payload)
                if event in ("generating", "complete"):
                    outputs.append(payload)
                if event == "complete":
                    break
    if not outputs or not outputs[-1] or outputs[-1][0].get("phase") != "complete":
        raise RuntimeError("Missing prediction output")
    return {"latency": time.monotonic() - start, "outputs": outputs}


def burst():
    stop = threading.Event()
    snapshots = []

    def monitor():
        while not stop.is_set():
            try:
                snapshots.append(
                    {
                        "time": time.time(),
                        **requests.get(
                            args.url + "/metrics", headers=headers, timeout=10
                        ).json(),
                    }
                )
            except requests.RequestException as e:
                snapshots.append({"error": str(e)})
            stop.wait(0.5)

    thread = threading.Thread(target=monitor)
    thread.start()
    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            results = list(executor.map(call, range(args.requests)))
    finally:
        stop.set()
        thread.join()
    return {
        "seconds": time.monotonic() - started,
        "results": results,
        "snapshots": snapshots,
    }


report = {"url": args.url, "workload": args.workload, "runs": []}
for auto in (False, True):
    r = requests.post(
        args.url + "/configure",
        json={"workload": args.workload, "autoscale": auto},
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    warm = [call() for _ in range(3)]
    run = {"autoscale": auto, "warm": warm, **burst()}
    while requests.get(args.url + "/metrics", headers=headers, timeout=10).json()[
        "starting"
    ]:
        time.sleep(1)
    run["steady"] = burst()
    report["runs"].append(run)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "autoscale": auto,
                "seconds": run["seconds"],
                "max_workers": max(len(s.get("workers", [])) for s in run["snapshots"]),
                "steady_seconds": run["steady"]["seconds"],
            }
        ),
        flush=True,
    )
    # Let an in-flight scale-up finish before switching workloads.
    while requests.get(args.url + "/metrics", headers=headers).json()["starting"]:
        time.sleep(1)
