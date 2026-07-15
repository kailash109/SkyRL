"""Non-colocated GRPO training on GSM8K with disk-delta weight sync, on Modal.

A real two-node SkyRL training job exercising the ``disk`` weight-sync backend
end to end, with W&B logging:

- **Inference node**: plain ``vllm serve`` (real HF weights) with SkyRL's
  ``NewInferenceWorkerWrap`` worker extension and
  ``--weight-transfer-config '{"backend": "disk"}'``, exposed via a Modal
  tunnel. All endpoints the trainer needs (``/inference/v1/generate``,
  ``/pause``, ``/resume``, ``/get_world_size``, ``/init_weight_transfer_engine``,
  ``/collective_rpc``) are native vLLM (dev mode).
- **Trainer node**: full SkyRL FSDP env (``uv sync --extra fsdp``, same
  pattern as ``examples/tinker/ppo/modal_run.py``), running
  ``skyrl.train.entrypoints.main_base`` with
  ``run_engines_locally=false`` + the tunnel URL as the external engine, and
  ``weight_sync_backend=disk``. Every training step publishes an xor+zstd
  delta to the shared modal.Volume; the inference host patches its local
  checkpoint (verifying per-tensor checksums) and reloads.

The two containers intentionally run *different* CUDA stacks (trainer: repo
lockfile cu129; inference: PyPI vLLM cu13) — disk transport has no NCCL
between the nodes, so their environments only need to agree on the delta
format.

Usage (from the SkyRL repo root):

    # W&B key comes from the `wandb-secret` Modal secret automatically
    # (modal secret create wandb-secret WANDB_API_KEY=...); a locally
    # exported WANDB_API_KEY overrides it. No key anywhere -> console logging.
    # optional: WANDB_ENTITY, WANDB_PROJECT (default: skyrl-disk-wsync)
    # optional: SKYRL_TRAINER_GPU / SKYRL_INFERENCE_GPU (default H200:2 each)
    # optional: SKYRL_WANDB_SECRET to use a different secret name

    # default: Qwen3.6-27B on H200:8 (trainer) + H200:4 (inference, TP=4)
    # a full bf16 sync would be ~54 GB; the delta ships a few GB per step
    modal run examples/train/gsm8k/modal_disk_wsync_train.py

    # quick smoke first: 3 training steps
    modal run examples/train/gsm8k/modal_disk_wsync_train.py --max-steps 3

    # Qwen3.6-35B-A3B (MoE; FSDP works but the Megatron recipe in
    # examples/train/megatron/ is the tuned path for this model)
    modal run examples/train/gsm8k/modal_disk_wsync_train.py \
        --model Qwen/Qwen3.6-35B-A3B --tp 4

    # cheap smoke on a small dense model
    SKYRL_TRAINER_GPU=H200:2 SKYRL_INFERENCE_GPU=H200:2 \
        modal run examples/train/gsm8k/modal_disk_wsync_train.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --tp 2 --max-steps 2

    # extra config overrides pass straight through to main_base
    modal run examples/train/gsm8k/modal_disk_wsync_train.py \
        --extra-args "trainer.train_batch_size=64 trainer.epochs=2"

    # the inference host keeps a full local checkpoint copy in container
    # scratch (~54 GB for 27B); if scheduling lands on a disk-tight host, set
    # SKYRL_INFERENCE_DISK_GB=200 to request explicit ephemeral disk.

The sender logs per-sync compression ("Published weight delta vN: X MB vs
Y MB raw (Zx)") and the trainer logs `timing/sync_weights` to wandb — those
two together are the delta-sync scorecard.
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import threading
import time
from typing import Optional

import modal

MODEL_DEFAULT = "Qwen/Qwen3.6-27B"
INFERENCE_TP = 4
TRAINER_GPU = os.environ.get("SKYRL_TRAINER_GPU", "H200:8")
INFERENCE_GPU = os.environ.get("SKYRL_INFERENCE_GPU", "H200:4")
TRAINER_NUM_GPUS = int(TRAINER_GPU.split(":")[1]) if ":" in TRAINER_GPU else 1

# Per-model trainer hyperparameters (micro batches sized for H200s at the
# recommended GPU counts below). GPU counts themselves are fixed at module
# import, so set them via env vars:
#
#   model                      SKYRL_TRAINER_GPU  SKYRL_INFERENCE_GPU  --tp
#   Qwen/Qwen3.6-27B (default) H200:8 (default)   H200:4 (default)     4
#   Qwen/Qwen3.6-35B-A3B (MoE) H200:8             H200:4               4
#   Qwen/Qwen2.5-1.5B-Instruct H200:2             H200:2               2   (cheap smoke)
#   Qwen/Qwen2.5-7B-Instruct   H200:4             H200:2               2
#
# At 27B a full bf16 sync is ~54 GB on the wire; the delta typically ships
# a few GB — this is where disk-delta sync is clearly worthwhile.
# NOTE: Qwen3.6-35B-A3B is MoE; this script trains with FSDP, which works but
# is slower than the Megatron recipe in examples/train/megatron/.
MODEL_PRESETS = {
    "35b-a3b": dict(micro_train=1, micro_fwd=2, mini_batch=32),
    "27b": dict(micro_train=1, micro_fwd=2, mini_batch=32),
    "14b": dict(micro_train=2, micro_fwd=4, mini_batch=32),
    "7b": dict(micro_train=4, micro_fwd=8, mini_batch=32),
    "1.5b": dict(micro_train=8, micro_fwd=16, mini_batch=32),
}


def _preset_for(model: str) -> dict:
    name = model.lower()
    for key, preset in MODEL_PRESETS.items():
        if key in name:
            return preset
    return MODEL_PRESETS["1.5b"]


# Optional explicit ephemeral disk (GiB) for the inference container — the
# host-local checkpoint copy needs ~model-size bytes of scratch (54 GB @ 27B).
INFERENCE_DISK_GB = int(os.environ.get("SKYRL_INFERENCE_DISK_GB", "0"))

VLLM_PORT = 8000
SHARED_DIR = "/shared"
DATA_DIR = "/root/data/gsm8k"
CKPT_DIR = "/root/ckpts"
HF_CACHE = "/root/.cache/huggingface"
REMOTE_REPO = "/root/SkyRL"
SHARED_VOLUME_NAME = "skyrl-disk-weight-sync-shared"
PRE_READ_HOOK = "skyrl_modal_wsync_hooks:reload_shared_volume"
KEEP_VERSIONS = 5  # prune older applied deltas to bound volume growth


def _find_repo_root() -> pathlib.Path:
    for start in (pathlib.Path(__file__).resolve(), pathlib.Path.cwd().resolve()):
        base = start if start.is_dir() else start.parent
        for candidate in [base, *base.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "skyrl").is_dir():
                return candidate
    raise RuntimeError("run this script from inside the SkyRL repository")


# Repo location only matters locally (to build the image mounts); containers
# re-import this module, where the repo lives at REMOTE_REPO (trainer image)
# or is absent entirely (inference image).
REPO_ROOT = _find_repo_root() if modal.is_local() else pathlib.Path(REMOTE_REPO)
HOOK_FILE = REPO_ROOT / "tests/backends/skyrl_train/gpu/modal/modal_volume_hooks.py"

app = modal.App("skyrl-gsm8k-disk-wsync")

# W&B credentials: injected from the Modal secret (create with
#   modal secret create wandb-secret WANDB_API_KEY=...
# or override the name via SKYRL_WANDB_SECRET). A locally-exported
# WANDB_API_KEY takes precedence over the secret.
wandb_secret = modal.Secret.from_name(os.environ.get("SKYRL_WANDB_SECRET", "wandb-secret"))

shared_volume = modal.Volume.from_name(SHARED_VOLUME_NAME, create_if_missing=True, version=2)
hf_cache = modal.Volume.from_name("skyrl-hf-cache-v2", create_if_missing=True, version=2)
data_volume = modal.Volume.from_name("skyrl-disk-wsync-data", create_if_missing=True, version=2)
ckpt_volume = modal.Volume.from_name("skyrl-disk-wsync-ckpts", create_if_missing=True, version=2)

# --- Inference image: lean vLLM + skyrl source (same as the weight-sync tests) ---
inference_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("vllm==0.23.0")
    .pip_install("zstandard>=0.22.0", "loguru", "httpx", "hf_transfer")
    .env({"VLLM_SERVER_DEV_MODE": "1", "PYTHONPATH": "/root", "SKYRL_WSYNC_VOLUME": SHARED_VOLUME_NAME})
    .add_local_file(HOOK_FILE, "/root/skyrl_modal_wsync_hooks.py")
    .add_local_python_source("skyrl")
)

# --- Trainer image: full repo + uv sync --extra fsdp (modal_run.py pattern) ---
trainer_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ca-certificates", "libnuma1", "numactl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}",
            "HF_HOME": HF_CACHE,
            "UV_LINK_MODE": "copy",
            "UV_PROJECT_ENVIRONMENT": f"{REMOTE_REPO}/.venv",
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        REMOTE_REPO,
        copy=True,
        ignore=[".venv", "**/__pycache__", ".git", "**/.pytest_cache", "**/node_modules"],
    )
    .workdir(REMOTE_REPO)
    .run_commands("uv sync --extra fsdp", gpu="any")  # some wheels probe CUDA during install
)

with inference_image.imports():
    import httpx


@app.function(
    image=inference_image,
    gpu=INFERENCE_GPU,
    volumes={SHARED_DIR: shared_volume, HF_CACHE: hf_cache},
    ephemeral_disk=INFERENCE_DISK_GB * 1024 or None,  # MiB; None = default
    timeout=24 * 3600,
)
def inference_server(q: modal.Queue, model: str, tp: int) -> None:
    """vLLM with REAL weights (the disk-delta base both sides seed from)."""
    cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(tp),
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "2048",
        "--worker-extension-cls",
        "skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap",
        "--weight-transfer-config",
        json.dumps({"backend": "disk"}),
    ]
    proc = subprocess.Popen(cmd)
    try:
        deadline = time.monotonic() + 1800  # real weight download + engine init
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
            result = q.get(partition="done")  # block until training finishes
            print(f"[inference] trainer reported: {result}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def _prune_deltas_forever(disk_dir: str) -> None:
    """Prune applied delta versions to bound shared-volume growth.

    send_chunks only returns after the server applied a version, so every
    version <= max(published) has been applied; keeping the last
    KEEP_VERSIONS is safe. (Trainer-restart replay needs the full chain —
    acceptable loss for a soak/training job; restart instead reuses ckpts.)
    """
    import re
    import shutil

    while True:
        time.sleep(60)
        try:
            versions = sorted(int(m.group(1)) for e in os.listdir(disk_dir) if (m := re.fullmatch(r"v(\d+)", e)))
            for v in versions[:-KEEP_VERSIONS]:
                shutil.rmtree(os.path.join(disk_dir, f"v{v}"), ignore_errors=True)
        except FileNotFoundError:
            pass


@app.function(
    image=trainer_image,
    gpu=TRAINER_GPU,
    volumes={SHARED_DIR: shared_volume, HF_CACHE: hf_cache, "/root/data": data_volume, CKPT_DIR: ckpt_volume},
    secrets=[wandb_secret],
    timeout=24 * 3600,
)
def trainer(
    server_url: str,
    disk_dir: str,
    model: str,
    tp: int,
    run_name: str,
    max_steps: Optional[int],
    extra_args: str,
    wandb_api_key: str,
    wandb_entity: Optional[str],
    wandb_project: str,
) -> None:
    preset = _preset_for(model)
    print(f"[trainer] model={model} preset={preset} trainer_gpus={TRAINER_NUM_GPUS}")

    # Local WANDB_API_KEY (forwarded via param) wins over the Modal secret
    # (already injected into os.environ by `secrets=[wandb_secret]`).
    wandb_api_key = wandb_api_key or os.environ.get("WANDB_API_KEY", "")
    if not wandb_api_key:
        print("[trainer] WARNING: no WANDB_API_KEY (param or wandb-secret) — logging to console")

    env = os.environ.copy()
    env.update({"HOME": "/root", "WANDB_API_KEY": wandb_api_key})
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity

    # ---- 1. GSM8K parquet prep (cached on the data volume) ----
    if not (os.path.exists(f"{DATA_DIR}/train.parquet") and os.path.exists(f"{DATA_DIR}/validation.parquet")):
        print("[trainer] preparing GSM8K parquet files")
        subprocess.run(
            [
                "uv",
                "run",
                "--extra",
                "fsdp",
                "python",
                "examples/train/gsm8k/gsm8k_dataset.py",
                "--output_dir",
                DATA_DIR,
            ],
            cwd=REMOTE_REPO,
            env=env,
            check=True,
        )

    # ---- 2. Background pruner for the shared delta dir ----
    threading.Thread(target=_prune_deltas_forever, args=(disk_dir,), daemon=True).start()

    # ---- 3. Launch main_base against the external engine ----
    cmd = [
        "uv",
        "run",
        "--extra",
        "fsdp",
        "-m",
        "skyrl.train.entrypoints.main_base",
        f"data.train_data=['{DATA_DIR}/train.parquet']",
        f"data.val_data=['{DATA_DIR}/validation.parquet']",
        "trainer.algorithm.advantage_estimator=grpo",
        f"trainer.policy.model.path={model}",
        "trainer.strategy=fsdp",
        # Non-colocated: trainer owns this container's GPUs; engines are external.
        "trainer.placement.colocate_all=false",
        f"trainer.placement.policy_num_gpus_per_node={TRAINER_NUM_GPUS}",
        f"trainer.placement.ref_num_gpus_per_node={TRAINER_NUM_GPUS}",
        # External inference over the tunnel (data + control plane on one URL).
        "generator.inference_engine.run_engines_locally=false",
        f"generator.inference_engine.external_proxy_url={server_url}",
        f'generator.inference_engine.external_server_urls=["{server_url}"]',
        "generator.inference_engine.num_engines=1",
        f"generator.inference_engine.tensor_parallel_size={tp}",
        # Disk-delta weight sync over the shared volume.
        "generator.inference_engine.weight_sync_backend=disk",
        f"generator.inference_engine.weight_sync_disk_dir={disk_dir}",
        f"generator.inference_engine.weight_sync_disk_pre_read_hook={PRE_READ_HOOK}",
        # Batch config from the model-size preset (H200-sized micro batches).
        "trainer.epochs=1",
        "trainer.train_batch_size=128",
        f"trainer.policy_mini_batch_size={preset['mini_batch']}",
        f"trainer.micro_forward_batch_size_per_gpu={preset['micro_fwd']}",
        f"trainer.micro_train_batch_size_per_gpu={preset['micro_train']}",
        "trainer.max_prompt_length=512",
        "generator.sampling_params.max_generate_length=1024",
        "generator.n_samples_per_prompt=5",
        "generator.batched=true",
        # Qwen3+ chat templates default to thinking mode; keep GSM8K rollouts
        # direct-answer. Harmless for templates without the flag (e.g. Qwen2.5).
        "generator.chat_template_kwargs.enable_thinking=false",
        "environment.env_class=gsm8k",
        "trainer.policy.optimizer_config.lr=1.0e-6",
        "trainer.algorithm.use_kl_loss=true",
        # Eval + checkpoints + logging.
        "trainer.eval_before_train=true",
        "trainer.eval_interval=5",
        "trainer.eval_batch_size=512",
        "trainer.ckpt_interval=25",
        f"trainer.ckpt_path={CKPT_DIR}/{run_name}",
        "trainer.resume_mode=null",
        "trainer.log_path=/tmp/skyrl-logs",
        f"trainer.logger={'wandb' if wandb_api_key else 'console'}",
        f"trainer.project_name={wandb_project}",
        f"trainer.run_name={run_name}",
    ]
    if max_steps is not None:
        cmd.append(f"trainer.max_training_steps={max_steps}")
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    print("[trainer] command:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd, cwd=REMOTE_REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[train] {line}", end="")
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"main_base exited with code {code}")
    print("[trainer] training finished successfully")


@app.local_entrypoint()
def main(
    model: str = MODEL_DEFAULT,
    tp: int = INFERENCE_TP,
    max_steps: Optional[int] = None,
    extra_args: str = "",
) -> None:
    # May be empty locally — the trainer container also gets WANDB_API_KEY
    # from the `wandb-secret` Modal secret and prefers the local value.
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT", "skyrl-disk-wsync")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"gsm8k-grpo-disk-wsync-{run_id}"
    disk_dir = f"{SHARED_DIR}/train-{run_id}"

    with modal.Queue.ephemeral() as q:
        server_call = inference_server.spawn(q, model, tp)
        print("[main] waiting for inference server tunnel (model load can take a while)...")
        server_url = None
        deadline = time.time() + 2400
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
        print(f"[main] engine at {server_url}; launching trainer ({TRAINER_GPU}) — run: {run_name}")
        try:
            trainer.remote(
                server_url,
                disk_dir,
                model,
                tp,
                run_name,
                max_steps,
                extra_args,
                wandb_api_key,
                wandb_entity,
                wandb_project,
            )
        finally:
            q.put("done", partition="done")
        server_call.get(timeout=180)
    print(f"[main] run complete: {run_name} (project={wandb_project})")
