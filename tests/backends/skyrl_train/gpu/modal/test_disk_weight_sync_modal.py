"""Two-node disk-delta weight sync test on Modal.

Emulates a non-colocated SkyRL deployment across two Modal containers:

- **Inference node** (2x GPU, TP=2): plain ``vllm serve`` with dummy-loaded
  weights, SkyRL's ``NewInferenceWorkerWrap`` worker extension, and
  ``--weight-transfer-config '{"backend": "disk"}'`` so vLLM's factory
  creates SkyRL's ``DiskWeightTransferEngine``. Reached from the trainer via
  a ``modal.forward`` tunnel (control plane = plain HTTP, like SkyRL's
  ``RemoteInferenceClient``).
- **Trainer node** (1x GPU): loads the real HF weights, wraps them in
  ``WeightChunk``s (as the FSDP/Megatron extractors would), and drives
  ``DiskWeightTransferSender.send_chunks`` twice — v1 = real weights,
  v2 = perturbed weights — publishing xor+zstd deltas to a shared
  ``modal.Volume`` (the "shared filesystem").

Checks:
1. Dummy weights produce gibberish (no "Paris").
2. After delta v1, the server answers "Paris" (weights fully synced through
   the delta path, TP=2 resharding included).
3. After delta v2 (small perturbation), the server still answers "Paris"
   and reports the delta compression ratio (the Phase-0 measurement).

Run (from the SkyRL repo root):
    modal run tests/backends/skyrl_train/gpu/modal/test_disk_weight_sync_modal.py
"""

import json
import pathlib
import subprocess
import time

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SHARED_DIR = "/shared"
VLLM_PORT = 8000
GPU_INFERENCE = "L4:2"
GPU_TRAINER = "L4:1"
SHARED_VOLUME_NAME = "skyrl-disk-weight-sync-shared"
# The inference workers refresh the volume mount before reading deltas
# (Modal Volume writes from another container need reload() to be seen).
PRE_READ_HOOK = "skyrl_modal_wsync_hooks:reload_shared_volume"

app = modal.App("skyrl-disk-weight-sync-test")

# The "shared filesystem" between trainer and inference nodes.
shared_volume = modal.Volume.from_name(SHARED_VOLUME_NAME, create_if_missing=True, version=2)
hf_cache = modal.Volume.from_name("skyrl-hf-cache-v2", create_if_missing=True, version=2)

# Lean image: vLLM 0.23.0 (PyPI wheel, CUDA 13 — needs the CUDA 13 devel base
# for nvcc/JIT kernels) + the few deps the skyrl weight-sync modules need. The
# skyrl package itself is added as local source — the lazy weight_sync/__init__
# keeps ray & co. out of this chain.
# PYTHONPATH=/root so the `vllm serve` subprocess (and its mp workers) can
# import the mounted skyrl package for --worker-extension-cls.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("vllm==0.23.0")
    .pip_install("zstandard>=0.22.0", "loguru", "httpx", "hf_transfer")
    .env({"VLLM_SERVER_DEV_MODE": "1", "PYTHONPATH": "/root", "SKYRL_WSYNC_VOLUME": SHARED_VOLUME_NAME})
    .add_local_file(
        pathlib.Path(__file__).parent / "modal_volume_hooks.py",
        "/root/skyrl_modal_wsync_hooks.py",
    )
    .add_local_python_source("skyrl")
)


with image.imports():
    import httpx


# ---------------------------------------------------------------------------
# Inference node
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_INFERENCE,
    volumes={SHARED_DIR: shared_volume, "/root/.cache/huggingface": hf_cache},
    timeout=3600,
)
def inference_server(q: modal.Queue) -> None:
    """Run vLLM (TP=2, dummy weights) until the trainer signals shutdown."""
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "bfloat16",
        "--load-format",
        "dummy",  # dummy weights: proves the sync actually delivered them
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.7",
        "--max-model-len",
        "2048",
        "--worker-extension-cls",
        "skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap",
        "--weight-transfer-config",
        json.dumps({"backend": "disk"}),
    ]
    proc = subprocess.Popen(cmd)
    try:
        # Wait for the server to come up.
        deadline = time.monotonic() + 900
        while True:
            try:
                if httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited early with code {proc.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("vllm serve did not become healthy in time")
            time.sleep(2)

        with modal.forward(VLLM_PORT) as tunnel:
            print(f"[inference] vLLM up, tunnel: {tunnel.url}")
            q.put(tunnel.url, partition="server_url")
            # Serve until the trainer reports completion.
            result = q.get(partition="done")
            print(f"[inference] trainer reported: {result}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


# ---------------------------------------------------------------------------
# Trainer node
# ---------------------------------------------------------------------------


class HttpControlPlane:
    """Minimal RemoteInferenceClient stand-in for one server URL.

    Speaks the same protocol: vLLM-native /init_weight_transfer_engine plus
    SkyRL's chunked lifecycle over dev-mode /collective_rpc.
    """

    def __init__(self, server_url: str):
        self._url = server_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(900.0))

    async def _post(self, path: str, payload: dict) -> None:
        resp = await self._client.post(f"{self._url}{path}", json=payload)
        assert resp.status_code == 200, f"{path} failed [{resp.status_code}]: {resp.text}"

    async def init_weight_update_communicator(self, init_info) -> None:
        await self._post("/init_weight_transfer_engine", {"init_info": init_info.to_api_payload()})

    async def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        await self._post(
            "/collective_rpc",
            {"method": "skyrl_start_weight_update", "kwargs": {"is_checkpoint_format": is_checkpoint_format}},
        )

    async def update_weights_disk(self, update_info: dict) -> None:
        await self._post("/collective_rpc", {"method": "update_weights_disk", "kwargs": {"update_info": update_info}})

    async def finish_weight_update(self) -> None:
        await self._post("/collective_rpc", {"method": "skyrl_finish_weight_update"})

    async def completion(self, prompt: str) -> str:
        resp = await self._client.post(
            f"{self._url}/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": 32, "temperature": 0.0},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["choices"][0]["text"]


@app.function(
    image=image,
    gpu=GPU_TRAINER,
    volumes={SHARED_DIR: shared_volume, "/root/.cache/huggingface": hf_cache},
    timeout=3600,
)
async def trainer(server_url: str, disk_dir: str) -> dict:
    import os

    import torch
    from transformers import AutoModelForCausalLM

    from skyrl.backends.skyrl_train.weight_sync import WeightChunk, delta_codec
    from skyrl.backends.skyrl_train.weight_sync.disk_strategy import (
        DiskInitInfo,
        DiskWeightTransferSender,
    )

    client = HttpControlPlane(server_url)
    results: dict = {"disk_dir": disk_dir}

    # ===== Step 1: dummy weights -> gibberish =====
    prompt = "What is the capital of France?"
    text_before = await client.completion(prompt)
    print(f"[trainer] dummy-weight output: {text_before!r}")
    assert "Paris" not in text_before, "dummy weights unexpectedly produced the correct answer"
    results["dummy_output"] = text_before

    # ===== Step 2: init the disk transfer engine on the server =====
    # model_path is the HF repo id: both sides resolve it independently via
    # their own HF caches (delta_codec.resolve_checkpoint_dir), so no full
    # checkpoint crosses the shared volume.
    init_info = DiskInitInfo(
        override_existing_receiver=True,
        disk_dir=disk_dir,
        encoding="xor",
        checksum_algo="adler32",
        model_dtype_str="bfloat16",
        model_path=MODEL,
        pre_read_hook=PRE_READ_HOOK,
    )
    await client.init_weight_update_communicator(init_info)
    print("[trainer] disk transfer engine initialized on server")

    # ===== Step 3: "training step 0" -> sync real weights as delta v1 =====
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    sender = DiskWeightTransferSender(init_info, client)

    def chunks():
        # One chunk per parameter, like FSDPWeightExtractor's simple path.
        for name, param in model.named_parameters():
            yield WeightChunk(
                names=[name],
                dtypes=[str(param.dtype)],
                shapes=[list(param.shape)],
                tensors=[param.data],
            )

    t0 = time.perf_counter()
    await sender.send_chunks(chunks())
    sync1_s = time.perf_counter() - t0
    print(f"[trainer] delta v1 published + applied in {sync1_s:.1f}s")

    text_v1 = await client.completion(prompt)
    print(f"[trainer] v1 output: {text_v1!r}")
    assert "Paris" in text_v1, f"weight sync failed - expected 'Paris', got {text_v1!r}"
    results["v1_output"] = text_v1
    results["v1_sync_seconds"] = sync1_s

    # ===== Step 4: "training step 1" -> perturb weights, sync delta v2 =====
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 1e-4)  # small optimizer-step-like update

    t0 = time.perf_counter()
    await sender.send_chunks(chunks())
    sync2_s = time.perf_counter() - t0
    print(f"[trainer] delta v2 published + applied in {sync2_s:.1f}s")

    text_v2 = await client.completion(prompt)
    print(f"[trainer] v2 output: {text_v2!r}")
    assert "Paris" in text_v2, f"post-perturbation sync failed - expected 'Paris', got {text_v2!r}"
    results["v2_output"] = text_v2
    results["v2_sync_seconds"] = sync2_s

    # ===== Step 5: delta compression report (Phase-0 measurement) =====
    raw_nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    for v in (1, 2):
        delta_bin = os.path.join(delta_codec.version_dir(disk_dir, v), delta_codec.DELTA_BIN_NAME)
        size = os.path.getsize(delta_bin)
        results[f"v{v}_delta_bytes"] = size
        results[f"v{v}_compression_ratio"] = raw_nbytes / size
        print(f"[trainer] delta v{v}: {size / 1e6:.1f} MB vs {raw_nbytes / 1e6:.1f} MB raw ({raw_nbytes / size:.2f}x)")
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    disk_dir = f"{SHARED_DIR}/deltas-{run_id}"  # fresh dir per run (delta-chain requirement)

    with modal.Queue.ephemeral() as q:
        server_call = inference_server.spawn(q)
        print("[main] waiting for inference server tunnel...")
        server_url = None
        deadline = time.time() + 1800
        while server_url is None:
            try:
                server_url = q.get(partition="server_url", timeout=15)
            except Exception:
                # Not up yet — check the server function didn't crash.
                try:
                    server_call.get(timeout=0)
                    raise RuntimeError("inference server exited before publishing its URL")
                except TimeoutError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError("timed out waiting for the inference server tunnel")
        print(f"[main] server at {server_url}; starting trainer")
        try:
            results = trainer.remote(server_url, disk_dir)
        finally:
            q.put("done", partition="done")
        server_call.get(timeout=120)

    print("\n========== DISK DELTA WEIGHT SYNC TEST PASSED ==========")
    for k, v in results.items():
        print(f"  {k}: {v}")
