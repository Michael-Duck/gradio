# Autoscaling PoC results — 2026-10-21

All eight cases completed with policy `poc-3`, using real HTTP/SSE requests and actual inference. The comparison is one persistent worker versus automatic scaling with a maximum of two workers. Each run has a scaling burst, then a second burst after any replica initialization finishes.

| Device | Workload | Requests per burst | One worker: steady seconds | Auto: steady seconds | Workers serving auto burst | Observed result |
|---|---|---:|---:|---:|---:|---|
| M5 32 GiB | python | 12 | 2.83 | 1.55 | 2 | 1.82× throughput |
| M5 32 GiB | kokoro | 64 | 33.22 | 19.03 | 2 | 1.75× throughput |
| M5 32 GiB | llama-2-7b.Q4_0.gguf | 6 | 16.21 | 13.83 | 1 | Kept one worker: memory headroom |
| M5 32 GiB | llama-2-7b.Q8_0.gguf | 6 | 22.72 | 23.08 | 1 | Kept one worker: memory headroom |
| A100 80 GiB | python | 12 | 12.62 | 6.43 | 2 | 1.96× throughput |
| A100 80 GiB | kokoro | 128 | 8.57 | 7.21 | 2 | 1.19× throughput |
| A100 80 GiB | Qwen/Qwen2.5-7B-Instruct | 12 | 18.44 | 12.79 | 2 | 1.44× throughput |
| A100 80 GiB | Qwen/Qwen2.5-32B-Instruct | 6 | 20.82 | 20.13 | 1 | Kept one worker: memory headroom |

These are small PoC measurements, not general performance guarantees. Timing differences in rows that retained one worker are run-to-run variation, not a scaling gain. Mac and A100 LLM tests use different models/runtimes; compare each auto run with its own one-worker baseline. Kokoro uses CPU on the Mac and CUDA on the A100.

## What the evidence confirms

- Every observed ready worker initialized its resource exactly once. Repeated requests reused the same worker/model; scaling created a separate model in a new PID.
- CPU and Kokoro requests used two serving processes on both devices. The A100 7B model also used two serving processes.
- The A100 32B worker peaked at roughly 66 GB of CUDA memory. The controller logged `Insufficient GPU memory headroom` and never attempted a second replica.
- The Mac Q4/Q8 workers peaked around 4.6/8.0 GB of process/unified memory. With other applications running, the reserve check rejected a second replica. This is not a claim that a 32 GiB Mac can never hold two such models.
- All LLM results contained exactly 64 generated tokens. Kokoro returned nonempty audio sample counts, and CPU results matched the expected checksum.
- No OOM occurred in the completed test matrix.

## Startup costs matter

- Mac, kokoro: the scaling burst took 22.09s versus 32.25s with one worker. The warmed-up burst took 19.03s versus 33.22s. A short burst can finish before a replica is ready, or be slower while the second model loads.
- Mac, python: the scaling burst took 2.19s versus 2.86s with one worker. The warmed-up burst took 1.55s versus 2.83s. A short burst can finish before a replica is ready, or be slower while the second model loads.
- A100, kokoro: the scaling burst took 10.26s versus 8.50s with one worker. The warmed-up burst took 7.21s versus 8.57s. A short burst can finish before a replica is ready, or be slower while the second model loads.
- A100, medium: the scaling burst took 18.55s versus 18.48s with one worker. The warmed-up burst took 12.79s versus 18.44s. A short burst can finish before a replica is ready, or be slower while the second model loads.
- A100, python: the scaling burst took 9.52s versus 12.54s with one worker. The warmed-up burst took 6.43s versus 12.62s. A short burst can finish before a replica is ready, or be slower while the second model loads.

The policy excludes the first request from the first-worker speed baseline and warms the new worker before its throughput trial. It can restart a trial on a later sustained burst. It keeps a replica only after observing a throughput improvement under continuous demand; idle or memory-pressure retirement remains enabled.

## Executor overhead and checks

- Warm local no-op executor round trip: 0.23 ms median, 0.28 ms p95 over 500 requests. This excludes HTTP, queue waiting, model work, and large-payload serialization.
- `pytest test/test_processing.py test/test_server.py`: 15 passed, 1 skipped (existing test).
- Ruff and `git diff --check`: clean.
- Regression coverage includes persistent resources, synchronous/async/streaming calls, high concurrency without thread starvation, cancellation/recovery, startup failure, memory admission, memory-pressure draining, and idle retirement.

## Running demos and evidence

- Local demo: http://127.0.0.1:7869/
- Private A100 Space: https://huggingface.co/spaces/abidlabs/gradio-queue-autoscale-poc
- The Space remains on A100 hardware; it was not paused. Both demos are reset to the lightweight Python workload for interactive use.
- Remote reconfiguration uses the control secret saved on this machine at `/tmp/autoscale-control-token`. It is intentionally absent from source and results.
- Raw results: compressed `*.json.gz` files in [`results/`](results/). [`summary.json`](results/summary.json) includes PIDs, memory peaks, timing comparisons and scaling decisions. [`environment.json`](results/environment.json) records versions and hardware.
- Reproduction and API limitations: [`README.md`](README.md).

This is an experimental App/Server process executor. It is capped at two replicas and does not change existing Blocks queue defaults. Memory is observed, not hard-limited; unseen inputs and transient allocations can still cause OOM. Broader workload sampling, resource groups, serialization/session adapters and stronger lifecycle handling are needed before making this a general default.
