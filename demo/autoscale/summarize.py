"""Summarize benchmark JSON without treating a starting process as useful capacity."""

import argparse
import gzip
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")
parser.add_argument("--output", required=True)
args = parser.parse_args()
summary = []
for name in args.files:
    report = json.loads(
        gzip.decompress(Path(name).read_bytes())
        if name.endswith(".gz")
        else Path(name).read_text()
    )
    item = {
        "source": Path(name).name,
        "url": report["url"],
        "workload": report["workload"],
        "runs": [],
    }
    for run in report["runs"]:
        phases = [run, run.get("steady", run)]
        snapshots = [s for p in phases for s in p["snapshots"] if "workers" in s]
        outputs = [r["outputs"][-1][0] for p in phases for r in p["results"]]
        if not all(o["phase"] == "complete" for o in outputs):
            raise ValueError("Incomplete response")
        if report["workload"] in ("medium", "large") and not all(
            o["tokens"] == 64 for o in outputs
        ):
            raise ValueError("Unexpected token count")
        if report["workload"] == "kokoro" and not all(
            o["samples"] > 0 for o in outputs
        ):
            raise ValueError("Missing audio")
        if report["workload"] == "python" and not all(
            o["checksum"] == 1565696 for o in outputs
        ):
            raise ValueError("Incorrect CPU output")
        ready = [w for s in snapshots for w in s["workers"] if w["ready"]]
        if not all(w["resource_initializations"] == [1] for w in ready):
            raise ValueError("Resource initialized more than once")
        item["runs"].append(
            {
                "autoscale": run["autoscale"],
                "requests_per_burst": len(run["results"]),
                "ramp_seconds": run["seconds"],
                "steady_seconds": phases[1]["seconds"],
                "serving_pids": sorted({o["pid"] for o in outputs}),
                "steady_serving_pids": sorted(
                    {r["outputs"][-1][0]["pid"] for r in phases[1]["results"]}
                ),
                "max_observed_processes": max(len(s["workers"]) for s in snapshots),
                "peak_worker_ram_gib": max((w["peak_ram"] for w in ready), default=0)
                / 1024**3,
                "peak_worker_gpu_gib": max((w["peak_gpu"] for w in ready), default=0)
                / 1024**3,
                "resources_initialized_once": True,
                "last_events": snapshots[-1]["events"],
                "model": outputs[-1].get("model", report["workload"]),
            }
        )
    if len(item["runs"]) == 2:
        item["ramp_speedup"] = (
            item["runs"][0]["ramp_seconds"] / item["runs"][1]["ramp_seconds"]
        )
        item["steady_speedup"] = (
            item["runs"][0]["steady_seconds"] / item["runs"][1]["steady_seconds"]
        )
    summary.append(item)
Path(args.output).write_text(json.dumps(summary, indent=2))
