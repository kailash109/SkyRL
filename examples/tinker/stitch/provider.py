"""Multi-run Stitch sidecar for a LoRA-serving SGLang rollout replica."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skyrl.tinker.stitch import LoraBulletinBoard
from stitch.protocol import SyncState, VersionManifest, WeightVersionPolicy
from stitch.servers.sglang import create_app
from stitch.sync import CommitMode, RolloutAdmissionGate

logger = logging.getLogger(__name__)


@dataclass
class SGLangLoraAdapter:
    """Loads independently versioned PEFT adapters into one SGLang server."""

    upstream_url: str
    staging_dir: str = "/tmp/stitch-lora"
    backend: str = "lora"
    _loaded: dict[str, int] = field(default_factory=dict, init=False)

    def _client(self, timeout: float | None):
        import httpx

        return httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def flush_cache(self) -> None:
        async with self._client(120.0) as client:
            response = await client.get(f"{self.upstream_url}/flush_cache")
            if response.status_code not in (200, 404):
                response.raise_for_status()

    async def pause_generation(self) -> None:
        async with self._client(120.0) as client:
            response = await client.post(f"{self.upstream_url}/pause_generation", json={"mode": "in_place"})
            response.raise_for_status()

    async def continue_generation(self) -> None:
        async with self._client(120.0) as client:
            response = await client.post(f"{self.upstream_url}/continue_generation", json={})
            response.raise_for_status()

    async def apply_manifest(self, manifest: VersionManifest, version_path: str) -> None:
        run_id = str(manifest.run_id)
        load_path = await asyncio.to_thread(self._stage, run_id, version_path)
        async with self._client(None) as client:
            if run_id in self._loaded:
                response = await client.post(
                    f"{self.upstream_url}/unload_lora_adapter",
                    json={"lora_name": run_id},
                )
                response.raise_for_status()
                del self._loaded[run_id]
            response = await client.post(
                f"{self.upstream_url}/load_lora_adapter",
                json={"lora_name": run_id, "lora_path": load_path},
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("success") is False:
                raise RuntimeError(f"SGLang rejected LoRA {run_id} v{manifest.version}: {data}")
        self._loaded[run_id] = manifest.version
        await asyncio.to_thread(self._prune_staging, run_id, Path(load_path))

    async def unload(self, run_id: str) -> None:
        if run_id not in self._loaded:
            return
        async with self._client(120.0) as client:
            response = await client.post(
                f"{self.upstream_url}/unload_lora_adapter",
                json={"lora_name": run_id},
            )
            response.raise_for_status()
        del self._loaded[run_id]
        await asyncio.to_thread(shutil.rmtree, Path(self.staging_dir) / run_id, True)

    def _stage(self, run_id: str, version_path: str) -> str:
        source = Path(version_path)
        target = Path(self.staging_dir) / run_id / source.name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return str(target)

    def _prune_staging(self, run_id: str, keep: Path) -> None:
        for version_dir in (Path(self.staging_dir) / run_id).iterdir():
            if version_dir != keep:
                shutil.rmtree(version_dir, ignore_errors=True)


@dataclass
class ChainState:
    current_version: int = 0
    last_used: float = 0.0


class MultiRunLoraSyncManager(RolloutAdmissionGate):
    """Maintains a bounded cache of independently versioned LoRA chains."""

    current_run_id = None

    def __init__(
        self,
        board: LoraBulletinBoard,
        engine: SGLangLoraAdapter,
        max_hot_chains: int,
        commit_mode: CommitMode = "quiesce",
    ) -> None:
        super().__init__(commit_mode=commit_mode)
        self.board = board
        self.engine = engine
        self.max_hot_chains = max_hot_chains
        self.debug_requests = False
        self.chains: dict[str, ChainState] = {}
        self._sync_task: asyncio.Task[None] | None = None
        self._sync_lock = asyncio.Lock()

    def _version_for(self, run_id: str | None) -> int:
        if run_id is None:
            return 0
        chain = self.chains.get(run_id)
        if chain is None:
            return 0
        chain.last_used = time.monotonic()
        return chain.current_version

    def _policy_error(
        self,
        policy: WeightVersionPolicy,
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        if run_id is not None and run_id not in self.chains and not self.board.has_run(run_id):
            return {
                "error": {
                    "type": "WeightRunNotRegistered",
                    "run_id": run_id,
                    "message": f"run {run_id!r} is not registered",
                }
            }
        return super()._policy_error(policy, run_id)

    def _on_policy_violation(self, error: dict[str, Any], run_id: str | None = None) -> None:
        if error["error"]["type"] == "WeightVersionNotReady" and run_id is not None:
            self.queue_sync(run_id=run_id)

    def queue_sync(self, target_version: int | None = None, run_id: str | None = None) -> None:
        del target_version
        if run_id is not None:
            self.chains.setdefault(run_id, ChainState()).last_used = time.monotonic()
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.get_running_loop().create_task(self.sync_to())

    async def startup_sync(self) -> None:
        await self.sync_to()

    async def sync_to(self) -> None:
        try:
            await self._sync_once()
        except Exception:
            logger.exception("Stitch LoRA reconciliation failed")

    async def _sync_once(self) -> None:
        async with self._sync_lock:
            await self.board.refresh()
            selected = sorted(
                self.chains,
                key=lambda run_id: self.chains[run_id].last_used,
                reverse=True,
            )[: self.max_hot_chains]
            unloads = [
                run_id for run_id, chain in self.chains.items() if chain.current_version > 0 and run_id not in selected
            ]
            loads: dict[str, tuple[int, VersionManifest, Path]] = {}
            for run_id in selected:
                version = self.board.read_latest(run_id)
                if version > self.chains[run_id].current_version:
                    loads[run_id] = (
                        version,
                        self.board.read_manifest(run_id, version),
                        self.board.version_dir(run_id, version),
                    )
            if not unloads and not loads:
                return

            async def apply() -> None:
                if self.commit_mode != "in_place":
                    await self.engine.flush_cache()
                for run_id in unloads:
                    await self.engine.unload(run_id)
                for _, manifest, path in loads.values():
                    await self.engine.apply_manifest(manifest, str(path))

            def applied() -> None:
                for run_id in unloads:
                    self.chains.pop(run_id, None)
                for run_id, (version, _, _) in loads.items():
                    self.chains[run_id].current_version = version

            await self.commit_version(
                apply=apply,
                on_applied=applied,
                pause=self.engine.pause_generation,
                resume=self.engine.continue_generation,
            )

    @property
    def current_version(self) -> int:
        return 0

    @property
    def sync_state(self) -> SyncState:
        if self._committing:
            return SyncState.COMMITTING
        if self._sync_task is not None and not self._sync_task.done():
            return SyncState.PREPARING
        return SyncState.IDLE

    async def server_info(self) -> dict[str, Any]:
        return {
            "backend": self.engine.backend,
            "sync_state": self.sync_state.value,
            "chains": {
                run_id: {"current_version": chain.current_version} for run_id, chain in sorted(self.chains.items())
            },
        }


def build_app(
    upstream_url: str,
    bulletin_root: str,
    bulletin_volume: str,
    max_hot_chains: int,
):
    refresh = None
    if bulletin_volume:
        from stitch.providers.modal import volume_reloader

        refresh = volume_reloader(bulletin_volume)
    manager = MultiRunLoraSyncManager(
        LoraBulletinBoard(bulletin_root, refresh=refresh),
        SGLangLoraAdapter(upstream_url),
        max_hot_chains=max_hot_chains,
    )
    return create_app(
        manager,
        upstream_url=upstream_url,
        run_resolver=lambda payload: payload.get("lora_path"),
        background_sync_interval=5.0,
    )


def start_sglang(model: str, port: int, max_loras: int, max_lora_rank: int) -> subprocess.Popen:
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enable-lora",
        "--max-loras-per-batch",
        str(max_loras),
        "--max-lora-rank",
        str(max_lora_rank),
        "--lora-target-modules",
        "all",
        "--mem-fraction-static",
        "0.8",
    ]
    return subprocess.Popen(cmd, start_new_session=True)


def wait_http(url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited while waiting for {url}: {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8001")
    parser.add_argument("--bulletin-root", required=True)
    parser.add_argument("--bulletin-volume", default="")
    parser.add_argument("--max-hot-chains", type=int, default=8)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        build_app(
            args.upstream_url,
            args.bulletin_root,
            args.bulletin_volume,
            args.max_hot_chains,
        ),
        host="0.0.0.0",
        port=args.port,
    )


if __name__ == "__main__":
    main()
