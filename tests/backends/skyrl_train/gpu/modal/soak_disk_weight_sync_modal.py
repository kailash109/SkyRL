"""Long-run soak test for disk-delta weight sync on Modal (two nodes).

Emulates many training steps of a non-colocated deployment:

- **Inference node** (2x GPU, TP=2): plain ``vllm serve`` (dummy-loaded) with
  SkyRL's worker extension and the ``disk`` weight-transfer backend.
- **Trainer node** (1x GPU): runs N *real* AdamW steps (causal-LM loss on
  random batches — realistic update magnitudes, hence realistic delta
  compression), publishing one xor+zstd delta per step to a shared
  modal.Volume and driving the apply over HTTP.

Long-run correctness is enforced three ways:
1. Per-sync: the inference host verifies a post-apply checksum for every
   tensor (any base drift or corruption is a hard 500, failing the run).
2. End-of-run: the trainer independently replays the FULL published delta
   chain from the shared volume onto a fresh checkpoint state and compares
   byte-for-byte against its live weights — validating the whole chain,
   not just the last sync.
3. Periodic generation probes confirm the server stays healthy/coherent.

Run (from the SkyRL repo root):
    modal run tests/backends/skyrl_train/gpu/modal/soak_disk_weight_sync_modal.py --steps 30

Notes:
- Each step publishes a delta (hundreds of MB for 0.5B at realistic lr);
  applied versions older than the last 3 are pruned to bound volume growth.
  (Pruning breaks trainer-restart replay — fine for a soak run.)
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
PRE_READ_HOOK = "skyrl_modal_wsync_hooks:reload_shared_volume"
PROBE_EVERY = 5  # generation probe every N steps
KEEP_VERSIONS = 3  # prune applied delta versions older than this many

app = modal.App("skyrl-disk-weight-sync-soak")

shared_volume = modal.Volume.from_name(SHARED_VOLUME_NAME, create_if_missing=True, version=2)
hf_cache = modal.Volume.from_name("skyrl-hf-cache-v2", create_if_missing=True, version=2)

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


@app.function(
    image=image,
    gpu=GPU_INFERENCE,
    volumes={SHARED_DIR: shared_volume, "/root/.cache/huggingface": hf_cache},
    timeout=4 * 3600,
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
        "dummy",
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
            result = q.get(partition="done")
            print(f"[inference] trainer reported: {result}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


class HttpControlPlane:
    """Minimal RemoteInferenceClient stand-in for one server URL."""

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

    async def completion(self, prompt: str, max_tokens: int = 24) -> str:
        resp = await self._client.post(
            f"{self._url}/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["choices"][0]["text"]


@app.function(
    image=image,
    gpu=GPU_TRAINER,
    volumes={SHARED_DIR: shared_volume, "/root/.cache/huggingface": hf_cache},
    timeout=4 * 3600,
)
async def trainer(server_url: str, disk_dir: str, steps: int, prune: bool) -> dict:
    import os
    import shutil

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from skyrl.backends.skyrl_train.weight_sync import WeightChunk, delta_codec
    from skyrl.backends.skyrl_train.weight_sync.disk_strategy import (
        DiskInitInfo,
        DiskWeightTransferSender,
    )

    client = HttpControlPlane(server_url)
    prompt = "What is the capital of France?"

    # ===== Setup: init receiver, sync initial weights (v1) =====
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

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    vocab_size = model.config.vocab_size
    sender = DiskWeightTransferSender(init_info, client)

    def chunks():
        for name, param in model.named_parameters():
            yield WeightChunk(names=[name], dtypes=[str(param.dtype)], shapes=[list(param.shape)], tensors=[param.data])

    def train_step() -> float:
        """One real causal-LM AdamW step on a random batch."""
        input_ids = torch.randint(0, vocab_size, (2, 256), device="cuda")
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(out.loss.detach())

    sync_times, delta_sizes = [], []
    raw_nbytes = sum(p.numel() * p.element_size() for p in model.parameters())

    async def sync(version: int) -> None:
        t0 = time.perf_counter()
        await sender.send_chunks(chunks())
        sync_times.append(time.perf_counter() - t0)
        delta_bin = os.path.join(delta_codec.version_dir(disk_dir, version), delta_codec.DELTA_BIN_NAME)
        delta_sizes.append(os.path.getsize(delta_bin))

    await sync(version=1)
    text = await client.completion(prompt)
    print(f"[trainer] v1 (pretrained weights) output: {text!r}")
    assert "Paris" in text, f"initial sync failed: {text!r}"

    # ===== Soak loop: real optimizer step -> delta sync, N times =====
    for step in range(1, steps + 1):
        loss = train_step()
        await sync(version=step + 1)
        msg = (
            f"[trainer] step {step}/{steps}: loss={loss:.3f} "
            f"delta={delta_sizes[-1] / 1e6:.1f}MB ({raw_nbytes / delta_sizes[-1]:.2f}x) "
            f"sync={sync_times[-1]:.1f}s"
        )
        if step % PROBE_EVERY == 0 or step == steps:
            probe = await client.completion(prompt)
            msg += f" probe={probe[:48]!r}"
        print(msg)

        if prune:
            # Prune old applied versions to bound volume growth (both sides
            # have applied version step+1 once sync() returned).
            for old in range(1, step + 1 - KEEP_VERSIONS + 1):
                shutil.rmtree(delta_codec.version_dir(disk_dir, old), ignore_errors=True)

    # ===== End-of-run chain verification =====
    # The remote enforces per-tensor post-apply checksums on EVERY sync, so
    # the inference host's local checkpoint provably matched the trainer's
    # live weights after each step (transitively, for the whole chain).
    # Here we add local checks:
    # 1. Sender base snapshot == live weights (the invariant that keeps the
    #    next delta correct).
    mismatches = []
    for name, param in model.named_parameters():
        expected = param.data.detach().contiguous().view(torch.uint8).cpu().numpy().reshape(-1)
        actual = sender._base.get(name)
        if actual is None or not np.array_equal(actual, expected):
            mismatches.append(name)
    assert not mismatches, f"base snapshot diverged from live weights for: {mismatches[:5]}"

    # 2. With pruning disabled: independently replay the FULL published chain
    #    from the shared volume onto a fresh checkpoint state and byte-compare
    #    against the live model — end-to-end validation of every delta file.
    if not prune:
        state = delta_codec.load_checkpoint_state(MODEL, "bfloat16")
        for v in sorted(delta_codec.list_versions(disk_dir)):
            delta_codec.apply_delta_to_state(delta_codec.version_dir(disk_dir, v), state)
        replay_mismatches = []
        for name, param in model.named_parameters():
            expected = param.data.detach().contiguous().view(torch.uint8).cpu().numpy().reshape(-1)
            if not np.array_equal(state.get(name), expected):
                replay_mismatches.append(name)
        assert not replay_mismatches, f"full-chain replay diverged for: {replay_mismatches[:5]}"
        print(f"[trainer] full-chain replay of {steps + 1} versions is byte-identical to live weights")

    final = await client.completion(prompt)
    print(f"[trainer] final server output after {steps} trained syncs: {final!r}")

    # Clean up this run's delta dir on the shared volume.
    shutil.rmtree(disk_dir, ignore_errors=True)

    return {
        "steps": steps,
        "total_syncs": steps + 1,
        "raw_model_mb": raw_nbytes / 1e6,
        "mean_delta_mb": sum(delta_sizes[1:]) / max(1, len(delta_sizes) - 1) / 1e6,  # skip v1 (zero delta)
        "mean_compression_x": raw_nbytes * (len(delta_sizes) - 1) / max(1, sum(delta_sizes[1:])),
        "mean_sync_s": sum(sync_times) / len(sync_times),
        "max_sync_s": max(sync_times),
        "final_output": final,
    }


@app.local_entrypoint()
def main(steps: int = 30, prune: bool = True):
    run_id = time.strftime("%Y%m%d-%H%M%S")
    disk_dir = f"{SHARED_DIR}/soak-{run_id}"

    with modal.Queue.ephemeral() as q:
        server_call = inference_server.spawn(q)
        print("[main] waiting for inference server tunnel...")
        server_url = None
        deadline = time.time() + 1800
        while server_url is None:
            try:
                server_url = q.get(partition="server_url", timeout=15)
            except Exception:
                try:
                    server_call.get(timeout=0)
                    raise RuntimeError("inference server exited before publishing its URL")
                except TimeoutError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError("timed out waiting for the inference server tunnel")
        print(f"[main] server at {server_url}; starting {steps}-step soak (prune={prune})")
        try:
            results = trainer.remote(server_url, disk_dir, steps, prune)
        finally:
            q.put("done", partition="done")
        server_call.get(timeout=120)

    print(f"\n========== DISK DELTA SOAK PASSED ({steps} steps) ==========")
    for k, v in results.items():
        print(f"  {k}: {v}")
