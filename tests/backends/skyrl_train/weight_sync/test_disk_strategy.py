"""CPU tests for the disk-based delta weight transfer strategy (sender side).

Run:
    uv run --extra dev pytest tests/backends/skyrl_train/weight_sync/test_disk_strategy.py -v

These tests must not import vLLM or NewInferenceWorkerWrap.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync import (
    DiskInitInfo,
    DiskTransferStrategy,
    DiskWeightTransferSender,
    WeightChunk,
    get_transfer_strategy,
    get_transfer_strategy_cls,
)
from skyrl.backends.skyrl_train.weight_sync import delta_codec as dc


@dataclass
class FakeInferenceClient:
    """Records the control-plane calls the sender makes."""

    calls: List[Any] = field(default_factory=list)

    async def start_weight_update(self, is_checkpoint_format: bool = True):
        self.calls.append(("start", is_checkpoint_format))
        return {}

    async def update_weights_disk(self, update_info: Dict[str, Any]):
        self.calls.append(("update", update_info))
        return {}

    async def finish_weight_update(self):
        self.calls.append(("finish",))
        return {}


def _tensor_bytes(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.uint8).numpy().reshape(-1)


def _make_model(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "model.embed.weight": torch.randn(32, 16, generator=g).to(torch.bfloat16),
        "model.layers.0.mlp.weight": torch.randn(64, 32, generator=g).to(torch.bfloat16),
        "model.norm.weight": torch.randn(16, generator=g).to(torch.bfloat16),
    }


def _write_checkpoint(ckpt_dir, tensors):
    dc.write_safetensors_file(
        str(ckpt_dir / "model.safetensors"),
        {n: ("bfloat16", list(t.shape), _tensor_bytes(t).tobytes()) for n, t in tensors.items()},
    )


def _chunks(tensors: Dict[str, torch.Tensor]):
    """One WeightChunk per tensor, matching extractor output."""
    for name, t in tensors.items():
        yield WeightChunk(
            names=[name],
            dtypes=[str(t.dtype)],
            shapes=[list(t.shape)],
            tensors=[t],
        )


def _init_info(tmp_path, **overrides) -> DiskInitInfo:
    kwargs = dict(
        override_existing_receiver=True,
        disk_dir=str(tmp_path / "deltas"),
        encoding="xor",
        checksum_algo="adler32",
        model_dtype_str="bfloat16",
        model_path=str(tmp_path / "ckpt"),
    )
    kwargs.update(overrides)
    return DiskInitInfo(**kwargs)


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def test_get_transfer_strategy_disk():
    assert get_transfer_strategy("disk", colocate_all=False) == "disk"
    assert get_transfer_strategy_cls("disk", colocate_all=False) is DiskTransferStrategy


def test_disk_strategy_rejects_colocated():
    with pytest.raises(ValueError, match="non-colocated"):
        get_transfer_strategy("disk", colocate_all=True)


def test_create_init_info_requires_disk_dir():
    from skyrl.train.config.config import InferenceEngineConfig

    ie_cfg = InferenceEngineConfig(weight_sync_backend="disk")
    with pytest.raises(ValueError, match="weight_sync_disk_dir"):
        DiskTransferStrategy.create_init_info(ie_cfg)


def test_create_init_info_from_config(tmp_path):
    from skyrl.train.config.config import InferenceEngineConfig

    ie_cfg = InferenceEngineConfig(
        weight_sync_backend="disk",
        weight_sync_disk_dir=str(tmp_path / "deltas"),
        weight_sync_delta_encoding="overwrite",
        weight_sync_delta_checksum="adler32",
        model_dtype="bfloat16",
    )
    info = DiskTransferStrategy.create_init_info(ie_cfg)
    assert info.disk_dir == str(tmp_path / "deltas")
    assert info.encoding == "overwrite"
    payload = info.to_api_payload()
    # payload keys must match DiskWeightTransferInitInfo fields on the engine side
    assert set(payload) == {
        "disk_dir",
        "encoding",
        "checksum_algo",
        "model_dtype",
        "local_checkpoint_dir",
        "model_path",
        "pre_read_hook",
    }
    assert info.for_servers(2, 3)[1].disk_dir == info.disk_dir


# ---------------------------------------------------------------------------
# Sender end-to-end (against an in-memory receiver via apply_delta_to_state)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
async def test_sender_publishes_versions(tmp_path, encoding):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    v0 = _make_model(seed=0)
    _write_checkpoint(ckpt_dir, v0)

    client = FakeInferenceClient()
    sender = DiskWeightTransferSender(_init_info(tmp_path, encoding=encoding), client)

    v1 = {n: (t + 0.01).to(t.dtype) for n, t in v0.items()}
    await sender.send_chunks(_chunks(v1))
    v2 = {n: (t + 0.02).to(t.dtype) for n, t in v1.items()}
    await sender.send_chunks(_chunks(v2))

    disk_dir = str(tmp_path / "deltas")
    assert dc.list_versions(disk_dir) == [1, 2]

    # control plane: start/update/finish per sync, with increasing versions
    kinds = [c[0] for c in client.calls]
    assert kinds == ["start", "update", "finish", "start", "update", "finish"]
    versions = [c[1]["version"] for c in client.calls if c[0] == "update"]
    assert versions == [1, 2]
    update_info = client.calls[1][1]
    assert set(update_info["names"]) == set(v0)
    assert all(d == "bfloat16" for d in update_info["dtype_names"])

    # receiver simulation: seed from the same checkpoint, replay both versions
    state = dc.load_checkpoint_state(str(ckpt_dir), "bfloat16")
    for v in (1, 2):
        dc.apply_delta_to_state(dc.version_dir(disk_dir, v), state)
    for name, t in v2.items():
        assert np.array_equal(state[name], _tensor_bytes(t)), name


@pytest.mark.asyncio
async def test_sender_reports_sync_stats(tmp_path):
    """get_last_sync_stats exposes per-sync transfer sizes for trainer metrics."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    v0 = _make_model(seed=0)
    _write_checkpoint(ckpt_dir, v0)

    sender = DiskWeightTransferSender(_init_info(tmp_path), FakeInferenceClient())
    assert sender.get_last_sync_stats() is None  # nothing synced yet

    await sender.send_chunks(_chunks({n: (t + 0.01).to(t.dtype) for n, t in v0.items()}))
    stats = sender.get_last_sync_stats()
    assert stats is not None
    assert stats["weight_sync/version"] == 1.0
    raw_nbytes = sum(t.numel() * t.element_size() for t in v0.values())
    assert stats["weight_sync/raw_mb"] == pytest.approx(raw_nbytes / 1e6)
    assert 0 < stats["weight_sync/delta_mb"] <= stats["weight_sync/raw_mb"]
    assert stats["weight_sync/compression_x"] >= 1.0


@pytest.mark.asyncio
async def test_sender_resume_replays_published_versions(tmp_path):
    """A restarted trainer rebuilds its base from disk and continues the chain."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    v0 = _make_model(seed=0)
    _write_checkpoint(ckpt_dir, v0)

    client = FakeInferenceClient()
    sender = DiskWeightTransferSender(_init_info(tmp_path), client)
    v1 = {n: (t + 0.01).to(t.dtype) for n, t in v0.items()}
    await sender.send_chunks(_chunks(v1))

    # "restart": fresh sender, same disk_dir
    sender2 = DiskWeightTransferSender(_init_info(tmp_path), FakeInferenceClient())
    v2 = {n: (t + 0.02).to(t.dtype) for n, t in v1.items()}
    await sender2.send_chunks(_chunks(v2))

    disk_dir = str(tmp_path / "deltas")
    assert dc.list_versions(disk_dir) == [1, 2]

    state = dc.load_checkpoint_state(str(ckpt_dir), "bfloat16")
    for v in (1, 2):
        dc.apply_delta_to_state(dc.version_dir(disk_dir, v), state)
    for name, t in v2.items():
        assert np.array_equal(state[name], _tensor_bytes(t)), name


@pytest.mark.asyncio
async def test_sender_full_record_for_tensor_missing_from_checkpoint(tmp_path):
    """Extractor names absent from the checkpoint (e.g. tied lm_head) ship as full records."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    v0 = _make_model(seed=0)
    _write_checkpoint(ckpt_dir, v0)

    extra_name = "lm_head.weight"
    weights = dict(v0)
    weights[extra_name] = torch.randn(8, 4).to(torch.bfloat16)

    sender = DiskWeightTransferSender(_init_info(tmp_path), FakeInferenceClient())
    await sender.send_chunks(_chunks(weights))

    disk_dir = str(tmp_path / "deltas")
    manifest = dc.read_manifest(dc.version_dir(disk_dir, 1))
    encodings = {t.name: t.encoding for t in manifest.tensors}
    assert encodings[extra_name] == dc.ENCODING_FULL
    assert all(enc == dc.ENCODING_XOR for name, enc in encodings.items() if name != extra_name)

    # second sync: now diffs against the (in-memory) base for the extra tensor too
    weights2 = {n: (t + 0.5).to(t.dtype) for n, t in weights.items()}
    await sender.send_chunks(_chunks(weights2))
    manifest2 = dc.read_manifest(dc.version_dir(disk_dir, 2))
    assert all(t.encoding == dc.ENCODING_XOR for t in manifest2.tensors)


@pytest.mark.asyncio
async def test_sender_rejects_dtype_mismatch(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    v0 = _make_model(seed=0)
    _write_checkpoint(ckpt_dir, v0)

    sender = DiskWeightTransferSender(_init_info(tmp_path), FakeInferenceClient())
    bad = {"model.norm.weight": torch.randn(16, dtype=torch.float32)}
    with pytest.raises(ValueError, match="single model dtype"):
        await sender.send_chunks(_chunks(bad))
    # aborted version must not be visible
    assert dc.list_versions(str(tmp_path / "deltas")) == []


@pytest.mark.asyncio
async def test_sender_delta_is_smaller_than_full(tmp_path):
    """Sanity: for a small perturbation, the published delta compresses well
    below the raw model bytes (the whole point of delta sync)."""
    import os

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    g = torch.Generator().manual_seed(0)
    v0 = {"w": torch.randn(256, 256, generator=g).to(torch.bfloat16)}
    _write_checkpoint(ckpt_dir, v0)

    sender = DiskWeightTransferSender(_init_info(tmp_path), FakeInferenceClient())
    # sparse update: only one row changes
    v1 = {"w": v0["w"].clone()}
    v1["w"][0, :] += 1.0
    await sender.send_chunks(_chunks(v1))

    delta_bin = os.path.join(dc.version_dir(str(tmp_path / "deltas"), 1), dc.DELTA_BIN_NAME)
    raw_nbytes = v0["w"].numel() * 2
    assert os.path.getsize(delta_bin) < raw_nbytes / 10
