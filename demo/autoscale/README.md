# Process queue autoscaling proof of concept

This branch adds `gr.App = gr.Server`, `@app.resource()`, and an experimental
`@app.queue()` process executor. Existing Blocks queues and `Server.api()` retain
their existing behavior. The experiment does **not** change all Gradio apps to
use processes by default.

```python
import gradio as gr
from collections.abc import Iterator

app = gr.App()


@app.resource()
def model():
    return load_model()


@app.queue()
def predict(prompt: str) -> Iterator[str]:
    yield from model().generate(prompt)


if __name__ == "__main__":
    app.launch()
```

The first request starts one persistent subprocess. Each resource initializes
once inside that subprocess, remains cached between requests, and initializes
independently in a second subprocess only when the scaler admits that replica.
No model object is copied from the HTTP process. Arbitrary Python, synchronous
and async functions, and streaming generators use the same executor.

After three completed requests beyond the first warmup request, sustained backlog
can trigger a second worker.
The admission check uses observed peak process memory, NVML GPU memory on NVIDIA,
and process physical footprint on macOS unified memory. It budgets another
observed replica plus 25%, with 15% of device memory reserved (at least 1 GiB).
Linux container RAM limits are read when available. This is an estimate, not an
OOM guarantee: future requests may need more memory than previously observed.

A six-completion trial under continuous demand compares aggregate throughput
and request latency with the one-worker baseline. The extra worker is retired
if throughput improves less than 10% or median service time exceeds 2.5 times
the baseline. The first request is excluded from the speed baseline to reduce initialization
bias. Idle replicas retire after 60 seconds, and depleted memory reserves drain
the extra replica after its current request finishes. Expansion cooldown is
60 seconds. Only two replicas are supported by this PoC. Set
`@app.queue(autoscale=False)` or `concurrency_limit=1` to retain one worker.

## Running the demo

Install this branch, `psutil`, `nvidia-ml-py` (CUDA), `torch`, `transformers`,
`accelerate`, `kokoro`, and `soundfile`. Kokoro also needs its English language
resources and espeak-ng. Then run `python demo/autoscale/app.py` and open
http://127.0.0.1:7869. Choose a workload, apply the configuration, warm it with
three requests, then submit a burst. The UI exposes replica PIDs, resource load
counts, memory peaks, and scaling decisions.

The benchmark uses the real HTTP/SSE API, verifies a nonempty output, and writes
raw request results and periodic worker telemetry:

```sh
python demo/autoscale/benchmark.py --workload python --output results.json
```

Use `--workload kokoro`, `medium`, or `large`. The model selection is demo code;
the scheduler does not inspect a model or require model metadata. Each LLM
request generates exactly 64 tokens. Remote control endpoints require the
`AUTOSCALE_CONTROL_TOKEN` secret; a private Space also requires HF authentication.

## Deliberate limits

- One application module, one GPU, one shared resource set and pool policy.
- Application top-level code runs in every worker. Put heavyweight initialization
  inside resource factories; avoid top-level side effects. Notebooks, nested
  application factories and imported endpoint modules need a loading contract.
- One request per worker. No tensor sharing, batching, distributed inference,
  framework patching, or automatic retries of user work.
- Resource factories are all preloaded in each worker. Lazy per-endpoint resource
  groups, teardown hooks and dependencies are future work.
- Cancellation retires the affected worker and its resources. A later request
  starts a replacement; the cancelled request is never replayed.
- Different inputs may change memory use and service time. The trial assumes a
  reasonably comparable workload; nonstationary traffic needs a better policy.
- Memory peaks are sampled every 250 ms; short spikes can be missed. NVML covers
  CUDA allocations rather than a particular ML framework. Other GPU backends,
  multiple GPUs, MIG and device remapping need dedicated providers.
- Persistent processes preserve state locally, but state is not shared between
  replicas. Stateful applications must explicitly account for this.
- This is an experiment, not a production-ready automatic OOM-prevention system.

On macOS, if `AUTOSCALE_GGUF_DIR` (default `/tmp/gradio-calibration-models`)
contains `llama-2-7b.Q4_0.gguf` and `llama-2-7b.Q8_0.gguf`, the medium/large
workloads use llama.cpp with all layers offloaded to Metal, instead of PyTorch.
Install `llama-cpp-python` with `CMAKE_ARGS='-DGGML_METAL=on'`. These existing
calibration files let us test a second GPU runtime without downloading another
20 GB of weights. The two precision variants exercise different memory sizes;
whether another replica fits still depends on live system memory.

For a private GPU Space, use the Dockerfile and metadata in `space/`, copy this
demo's `app.py` and `index.html` into the build root, and put wheels built from
this checkout and `client/python` in `wheels/`. Set the
`AUTOSCALE_CONTROL_TOKEN` Space secret, request `a100-large`, and run the benchmark
with `--url https://YOUR-SPACE.hf.space`. The benchmark obtains private-Space
authentication from the local Hugging Face login. CUDA workloads use Kokoro,
Qwen2.5-7B-Instruct FP16, and Qwen2.5-32B-Instruct FP16.

Run regression tests with `pytest test/test_processing.py test/test_server.py`.
The benchmark records a scaling burst and a second burst after initialization,
so startup costs remain visible alongside sustained performance.

Process arguments and results must be serializable. Live request/session objects,
open handles and process-local state need explicit adapters before this can be a
default for existing apps. Resource factories are synchronous; async factories
and cached clients tied to a particular event loop need more lifecycle work.
An initial resource-loading failure is reported to waiting requests; restart or
use the demo's reconfiguration control to retry initialization.
