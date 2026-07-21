"""Trainer-side publisher for Stitch LoRA sampler versions."""

from __future__ import annotations

from pathlib import Path

from skyrl.tinker.stitch.bulletin import ADAPTER_FILES, LoraBulletinBoard, lora_manifest


class StitchPublisher:
    """Publishes exported LoRA adapters to a shared bulletin board."""

    def __init__(self, root: str | Path, volume_name: str | None = None) -> None:
        self.board = LoraBulletinBoard(root)
        self.volume_name = volume_name

    def version_dir(self, model_id: str, version: int) -> Path:
        path = self.board.version_dir(model_id, version)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def publish(self, model_id: str, version: int, version_dir: str | Path) -> None:
        path = Path(version_dir)
        for name in ADAPTER_FILES:
            if not (path / name).is_file():
                raise FileNotFoundError(path / name)

        if self.board.read_latest(model_id) < version:
            self.board.publish_manifest(lora_manifest(model_id, version, path), path)

        if self.volume_name:
            from stitch.providers.modal import commit_volume

            commit_volume(self.volume_name)
