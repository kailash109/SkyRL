"""Run-partitioned bulletin board for LoRA sampler versions."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stitch.protocol import (
    Artifact,
    VersionManifest,
    atomic_write_text,
    decide_pointer_move,
    parse_weight_identity,
    weight_identity,
)

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


class LoraBulletinBoard:
    """Filesystem bulletin board with one monotonic pointer per model."""

    def __init__(self, root: str | Path, refresh: Callable[[], Any] | None = None) -> None:
        self.root = Path(root)
        self._refresh = refresh

    async def refresh(self) -> None:
        if self._refresh is None:
            return
        result = await asyncio.to_thread(self._refresh)
        if inspect.isawaitable(result):
            await result

    def _pointer_path(self, run_id: str) -> Path:
        return self.root / run_id / "latest"

    def has_run(self, run_id: str) -> bool:
        return self._pointer_path(run_id).exists()

    def read_latest(self, run_id: str) -> int:
        path = self._pointer_path(run_id)
        if not path.exists():
            return 0
        return parse_weight_identity(path.read_text(encoding="utf-8").strip()) or 0

    def advance(self, run_id: str, version: int) -> None:
        current = self.read_latest(run_id)
        decide_pointer_move(run_id, current, run_id=run_id, version=version)
        atomic_write_text(self._pointer_path(run_id), weight_identity(version))

    def version_dir(self, run_id: str, version: int) -> Path:
        return self.root / run_id / weight_identity(version)

    def read_manifest(self, run_id: str, version: int) -> VersionManifest:
        return VersionManifest.read(self.version_dir(run_id, version) / "manifest.json")

    def active_runs(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(entry.name for entry in self.root.iterdir() if entry.is_dir() and (entry / "latest").exists())

    def publish_manifest(self, manifest: VersionManifest, version_path: str | Path) -> None:
        manifest.write(Path(version_path) / "manifest.json")
        self.advance(str(manifest.run_id), manifest.version)


def lora_manifest(run_id: str, version: int, version_dir: str | Path) -> VersionManifest:
    """Build a manifest from the exported PEFT adapter configuration."""

    with (Path(version_dir) / "adapter_config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    return VersionManifest(
        version=version,
        base_version=0,
        backend="lora",
        load_format="peft",
        transition_files=list(ADAPTER_FILES),
        artifacts=[Artifact(kind="transition", path=name) for name in ADAPTER_FILES],
        run_id=run_id,
        base_model=config.get("base_model_name_or_path"),
        metadata={
            "trainer": "skyrl",
            "adapter": {
                "rank": config.get("r"),
                "alpha": config.get("lora_alpha"),
                "target_modules": config.get("target_modules", []),
            },
        },
    )
