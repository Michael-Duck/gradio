"""Run with python app.py. Same model code serves every replica."""

from collections.abc import Iterator
import hashlib
import os
import platform
from pathlib import Path
import time

import gradio as gr
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

app = gr.App()


@app.resource()
def model():
    workload = os.environ.get("AUTOSCALE_WORKLOAD", "python")
    if workload == "python":
        return {"created": time.time()}
    import torch

    torch.set_num_threads(1)
    if workload == "kokoro":
        from kokoro import KPipeline

        return KPipeline(
            lang_code="a", device="cuda" if torch.cuda.is_available() else "cpu"
        )
    local_models = Path(
        os.environ.get("AUTOSCALE_GGUF_DIR", "/tmp/gradio-calibration-models")
    )
    if platform.system() == "Darwin" and local_models.exists():
        import llama_cpp

        if not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError("Install llama-cpp-python with Metal support")
        filename = (
            "llama-2-7b.Q4_0.gguf" if workload == "medium" else "llama-2-7b.Q8_0.gguf"
        )
        loaded = llama_cpp.Llama(
            model_path=str(local_models / filename),
            n_gpu_layers=-1,
            n_ctx=512,
            n_threads=2,
            use_mmap=False,
            verbose=False,
        )
        return {"runtime": "llama.cpp/Metal", "loaded": loaded, "name": filename}
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("This workload requires a GPU; CPU fallback is disabled")
    # Model selection belongs to the demo, not the autoscaler.
    names = {
        "medium": "Qwen/Qwen2.5-7B-Instruct"
        if device == "cuda"
        else "Qwen/Qwen2.5-3B-Instruct",
        "large": "Qwen/Qwen2.5-32B-Instruct"
        if device == "cuda"
        else "Qwen/Qwen2.5-7B-Instruct",
    }
    revisions = {
        "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
        "Qwen/Qwen2.5-32B-Instruct": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    }
    name = names[workload]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revisions[name])
    loaded = AutoModelForCausalLM.from_pretrained(
        name, revision=revisions[name], dtype=torch.float16, device_map={"": device}
    )
    loaded.eval()
    return loaded, tokenizer, device, name


@app.queue(stream_every=0.1)
def predict(text: str = "Explain why the sky is blue.") -> Iterator[dict]:
    workload = os.environ.get("AUTOSCALE_WORKLOAD", "python")
    resource = model()
    started = time.monotonic()
    yield {"pid": os.getpid(), "phase": "running", "workload": workload}
    if workload == "python":
        # Pure Python integer arithmetic: deliberately CPU/GIL-bound, no sleeps.
        result = 0
        for i in range(8_000_000):
            result = (result + i * i) % 1_000_000_007
        output = {"checksum": result}
    elif workload == "kokoro":
        import numpy as np

        audio = np.concatenate(
            [audio.numpy() for _, _, audio in resource(text, voice="af_heart")]
        )
        output = {
            "samples": len(audio),
            "sample_rate": 24000,
            "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        }
    elif isinstance(resource, dict) and resource.get("runtime") == "llama.cpp/Metal":
        loaded = resource["loaded"]
        loaded.reset()
        result = loaded.create_completion(
            text,
            max_tokens=64,
            temperature=0,
            logit_bias={str(loaded.token_eos()): -1000},
        )
        output = {
            "model": resource["name"],
            "runtime": resource["runtime"],
            "tokens": result["usage"]["completion_tokens"],
            "text": result["choices"][0]["text"],
        }
        if output["tokens"] != 64:
            raise RuntimeError("Benchmark requires exactly 64 generated tokens")
    else:
        import torch

        loaded, tokenizer, device, name = resource
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = loaded.generate(
                **inputs,
                max_new_tokens=64,
                min_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize()
        output = {
            "model": name,
            "tokens": 64,
            "text": tokenizer.decode(
                ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            ),
        }
    yield {
        "pid": os.getpid(),
        "phase": "complete",
        "workload": workload,
        "seconds": time.monotonic() - started,
        **output,
    }


def authorize(request):
    secret = os.environ.get("AUTOSCALE_CONTROL_TOKEN")
    if secret:
        import secrets

        if not secrets.compare_digest(
            request.headers.get("x-control-token", ""), secret
        ):
            raise HTTPException(403, "Control token required")
    elif request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(403, "Local controls only")


@app.get("/metrics")
def metrics():
    return app._process_pool.status()


@app.post("/configure")
async def configure(request: Request):
    authorize(request)
    data = await request.json()
    if data.get("workload") not in {"python", "kokoro", "medium", "large"}:
        raise HTTPException(400, "Unknown workload")
    try:
        app._process_pool.configure(
            {"AUTOSCALE_WORKLOAD": data["workload"]},
            2 if data.get("autoscale", True) else 1,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()


if __name__ == "__main__":
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7869")),
        ssr_mode=False,
        show_error=True,
    )
