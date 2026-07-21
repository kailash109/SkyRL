"""Launch a Stitch SGLang rollout pool and SkyRL Tinker trainer on Modal.

Usage:
    uv run --isolated --extra tinker --with modal \
        modal run --detach examples/tinker/stitch/modal_run.py
"""

from __future__ import annotations

import json
import os
import pathlib

import modal
import modal.experimental

APP_NAME = "skyrl-tinker-stitch-gsm8k"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REMOTE_REPO = "/root/SkyRL"
HF_HOME = "/root/hf-cache"
BULLETIN_ROOT = "/bulletin"
BULLETIN_VOLUME = "skyrl-tinker-stitch-bulletin"
DATA_ROOT = "/root/data/gsm8k"
CHECKPOINT_ROOT = "/root/checkpoints"
SIDECAR_PORT = 8000
SGLANG_PORT = 8001


def _repo_root() -> pathlib.Path:
    for start in (pathlib.Path(__file__).resolve(), pathlib.Path.cwd().resolve()):
        candidate = start if start.is_dir() else start.parent
        for path in (candidate, *candidate.parents):
            if (path / "pyproject.toml").exists() and (path / "skyrl").exists():
                return path
    raise RuntimeError("Run the example from a SkyRL checkout")


REPO_ROOT = _repo_root()
STITCH_ROOT = pathlib.Path(os.environ.get("STITCH_LOCAL_DIR", "~/stitch")).expanduser()
bulletin_volume = modal.Volume.from_name(BULLETIN_VOLUME, create_if_missing=True, version=2)
hf_volume = modal.Volume.from_name("skyrl-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("skyrl-tinker-stitch-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("skyrl-tinker-stitch-checkpoints", create_if_missing=True)

training_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ca-certificates", "libnuma1", "numactl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}",
            "HF_HOME": HF_HOME,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/root/stitch/src",
            "UV_LINK_MODE": "copy",
            "UV_PROJECT_ENVIRONMENT": f"{REMOTE_REPO}/.venv",
        }
    )
    .add_local_dir(str(STITCH_ROOT), "/root/stitch", copy=True, ignore=[".git", "**/__pycache__"])
    .add_local_dir(str(REPO_ROOT), REMOTE_REPO, copy=True, ignore=[".venv", "**/__pycache__"])
    .workdir(REMOTE_REPO)
    .run_commands("uv sync --extra tinker --extra fsdp", gpu="any")
    .run_commands(f"rm -rf {HF_HOME}")
)

rollout_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.12")
    .entrypoint([])
    .env({"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(STITCH_ROOT), "/root/stitch", copy=True, ignore=[".git", "**/__pycache__"])
    .add_local_dir(str(REPO_ROOT), REMOTE_REPO, copy=True, ignore=[".venv", "**/__pycache__"])
    .run_commands("pip install -e /root/stitch", f"pip install --no-deps -e {REMOTE_REPO}")
    .run_commands(f"rm -rf {HF_HOME}")
    .workdir(REMOTE_REPO)
)

app = modal.App(APP_NAME)


@app.function(
    image=rollout_image,
    volumes={HF_HOME: hf_volume},
    timeout=60 * 60,
)
def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    hf_volume.commit()


@app.cls(
    image=rollout_image,
    gpu="H100:1",
    volumes={HF_HOME: hf_volume, BULLETIN_ROOT: bulletin_volume},
    min_containers=0,
    max_containers=4,
    scaledown_window=15 * 60,
    timeout=30 * 60,
)
@modal.experimental.http_server(
    port=SIDECAR_PORT,
    proxy_regions=["us-east"],
    startup_timeout=20 * 60,
)
@modal.concurrent(target_inputs=64)
class RolloutPool:
    """One SGLang server with a Stitch LoRA sidecar."""

    @modal.enter()
    def start(self) -> None:
        import subprocess

        from examples.tinker.stitch.provider import start_sglang, wait_http

        self.sglang = start_sglang(MODEL, SGLANG_PORT, max_loras=8, max_lora_rank=32)
        wait_http(f"http://127.0.0.1:{SGLANG_PORT}/health", self.sglang, 15 * 60)
        self.sidecar = subprocess.Popen(
            [
                "python3",
                "-m",
                "examples.tinker.stitch.provider",
                "--port",
                str(SIDECAR_PORT),
                "--upstream-url",
                f"http://127.0.0.1:{SGLANG_PORT}",
                "--bulletin-root",
                BULLETIN_ROOT,
                "--bulletin-volume",
                BULLETIN_VOLUME,
                "--max-hot-chains",
                "8",
            ],
            cwd=REMOTE_REPO,
            start_new_session=True,
        )
        wait_http(f"http://127.0.0.1:{SIDECAR_PORT}/health", self.sidecar, 5 * 60)

    @modal.exit()
    def stop(self) -> None:
        from examples.tinker.stitch.provider import terminate

        terminate(getattr(self, "sidecar", None))
        terminate(getattr(self, "sglang", None))


@app.function(
    image=training_image,
    gpu="H100:4",
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        HF_HOME: hf_volume,
        BULLETIN_ROOT: bulletin_volume,
        "/root/data": data_volume,
        CHECKPOINT_ROOT: checkpoint_volume,
    },
    timeout=24 * 60 * 60,
)
def train(rollout_url: str, steps: int = 2) -> None:
    import signal
    import subprocess
    import time
    import urllib.error
    import urllib.request

    os.makedirs(DATA_ROOT, exist_ok=True)
    train_data = f"{DATA_ROOT}/train.parquet"
    if not os.path.exists(train_data):
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "--extra",
                "tinker",
                "--extra",
                "fsdp",
                "examples/train/gsm8k/gsm8k_dataset.py",
                "--output_dir",
                DATA_ROOT,
            ],
            cwd=REMOTE_REPO,
            check=True,
        )
        data_volume.commit()

    backend_config = {
        "strategy": "fsdp",
        "trainer.placement.colocate_all": True,
        "trainer.placement.policy_num_gpus_per_node": 4,
        "trainer.micro_forward_batch_size_per_gpu": 4,
        "trainer.micro_train_batch_size_per_gpu": 4,
        "generator.inference_engine.num_engines": 4,
        "generator.inference_engine.tensor_parallel_size": 1,
    }
    server = subprocess.Popen(
        [
            "uv",
            "run",
            "--frozen",
            "--extra",
            "tinker",
            "--extra",
            "fsdp",
            "-m",
            "skyrl.tinker.api",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--base-model",
            MODEL,
            "--backend",
            "fsdp",
            "--backend-config",
            json.dumps(backend_config),
            "--checkpoints-base",
            CHECKPOINT_ROOT,
            "--external-inference-provider",
            "stitch",
            "--external-inference-url",
            rollout_url,
            "--stitch-bulletin-root",
            BULLETIN_ROOT,
            "--stitch-bulletin-volume",
            BULLETIN_VOLUME,
            "--stitch-max-retries",
            "1200",
        ],
        cwd=REMOTE_REPO,
        env={**os.environ, "TINKER_API_KEY": "tml-dummy"},
        start_new_session=True,
    )

    try:
        deadline = time.time() + 20 * 60
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Tinker server exited with {server.returncode}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/healthz", timeout=5):
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        else:
            raise TimeoutError("Tinker server did not become ready")

        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "--extra",
                "tinker",
                "--extra",
                "fsdp",
                "-m",
                "examples.tinker.stitch.grpo_client",
                "--base-url",
                "http://127.0.0.1:8000",
                "--model",
                MODEL,
                "--data",
                train_data,
                "--steps",
                str(steps),
            ],
            cwd=REMOTE_REPO,
            env={**os.environ, "TINKER_API_KEY": "tml-dummy"},
            check=True,
        )
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        checkpoint_volume.commit()


def rollout_gateway_url() -> str:
    urls = RolloutPool._experimental_get_flash_urls()
    if not urls:
        raise RuntimeError("RolloutPool has no Flash gateway URL")
    return str(urls[0]).rstrip("/")


@app.local_entrypoint()
def main(steps: int = 2, wait: bool = False) -> None:
    download_model.remote()
    rollout_url = rollout_gateway_url()
    if wait:
        train.remote(rollout_url, steps=steps)
    else:
        train.spawn(rollout_url, steps=steps)
        print("Submitted detached SkyRL training job")
