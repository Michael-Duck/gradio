"""Render the recorded PoC results; no inferred cross-device comparisons."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "results/summary.json").read_text())
order = [
    ("mac", "python"),
    ("mac", "kokoro"),
    ("gpu", "python"),
    ("gpu", "kokoro"),
    ("gpu", "medium"),
]
labels = [
    "M5 · Python",
    "M5 · Kokoro (CPU)",
    "A100 · Python",
    "A100 · Kokoro (CUDA)",
    "A100 · Qwen 7B (FP16)",
]
selected = [
    next(r for r in rows if f"-{device}-" in r["source"] and r["workload"] == workload)
    for device, workload in order
]
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"}
)
fig, ax = plt.subplots(figsize=(12, 7.8), facecolor="#fafafa")
ax.set_facecolor("#fafafa")
y = np.arange(len(selected))
for offset, field, color, label in [
    (-0.18, "ramp_speedup", "#a9b9cd", "Scale-up burst (includes replica startup)"),
    (0.18, "steady_speedup", "#e97725", "Warm burst (replica ready)"),
]:
    values = [r[field] for r in selected]
    ax.barh(y + offset, values, height=0.30, color=color, label=label, zorder=3)
    for pos, value in zip(y + offset, values):
        ax.text(value + 0.035, pos, f"{value:.2f}×", va="center", fontsize=11)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, 2.25)
ax.axvline(1, color="#5f6874", linestyle="--", linewidth=1.1, zorder=4)
ax.set_xlabel(
    "Throughput relative to one worker on the same device (higher is better)",
    labelpad=12,
)
ax.set_xticks([0, 0.5, 1, 1.5, 2], ["0×", "0.5×", "1×", "1.5×", "2×"])
ax.grid(axis="x", alpha=0.16, zorder=0)
ax.tick_params(axis="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False, fontsize=10)
fig.text(
    0.04,
    0.95,
    "Gradio process autoscaling: measured PoC results",
    fontsize=20,
    weight="bold",
    color="#212733",
)
fig.text(
    0.04,
    0.91,
    "Five workloads scaled from 1 → 2 workers; resources loaded once per worker.",
    fontsize=12,
    color="#515866",
)
fig.text(
    0.04,
    0.17,
    "Stayed at 1 worker: M5 Llama 2 7B Q4 / Q8 · A100 Qwen 32B FP16",
    weight="bold",
    fontsize=12,
)
fig.text(
    0.04,
    0.135,
    "Memory admission declined another replica under observed headroom. No OOM in these runs.",
    fontsize=11,
)
fig.text(
    0.04,
    0.075,
    "Fixed-input bursts, 6–128 requests; one measured run per configuration, not a performance guarantee.\nM5: 32 GiB unified memory. A100: 80 GiB VRAM. Ratios use each workload’s own baseline.\nSource: recorded HTTP/SSE experiments, policy poc-3 · 21 Oct 2026",
    fontsize=9,
    color="#626975",
    va="top",
    linespacing=1.5,
)
fig.subplots_adjust(left=0.25, right=0.94, top=0.79, bottom=0.29)
for suffix in ("png", "svg"):
    fig.savefig(
        ROOT / f"results/autoscale-results.{suffix}",
        dpi=180,
        facecolor=fig.get_facecolor(),
    )
