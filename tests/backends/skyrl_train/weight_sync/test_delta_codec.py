"""CPU tests for the disk-delta weight sync codec.

Run:
    uv run --extra dev pytest tests/backends/skyrl_train/weight_sync/test_delta_codec.py -v

These tests must not import vLLM or NewInferenceWorkerWrap.
"""

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync import delta_codec as dc


def _tensor_bytes(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.uint8).numpy().reshape(-1)


def _rand(shape, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype)


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_xor_roundtrip(dtype):
    base_t = _rand((64, 32), dtype, seed=1)
    new_t = base_t + 0.01 * _rand((64, 32), dtype, seed=2).to(dtype)
    base, new = _tensor_bytes(base_t), _tensor_bytes(new_t)

    delta = dc.xor_encode(new, base)
    buf = base.copy()
    dc.xor_apply(buf, delta)
    assert np.array_equal(buf, new)


def test_xor_is_involution():
    base = _tensor_bytes(_rand((16, 16), seed=3))
    new = _tensor_bytes(_rand((16, 16), seed=4))
    delta = dc.xor_encode(new, base)
    # applying twice returns to base
    buf = base.copy()
    dc.xor_apply(buf, delta)
    dc.xor_apply(buf, delta)
    assert np.array_equal(buf, base)


def test_xor_unchanged_is_zeros():
    base = _tensor_bytes(_rand((8, 8), seed=5))
    delta = dc.xor_encode(base.copy(), base)
    assert not delta.any()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_overwrite_roundtrip(dtype):
    itemsize = dc.TORCH_DTYPE_NBYTES[str(dtype).split(".")[-1]]
    base_t = _rand((32, 16), dtype, seed=6)
    new_t = base_t.clone()
    new_t[3, :] += 1.0  # change one row
    base, new = _tensor_bytes(base_t), _tensor_bytes(new_t)

    record = dc.overwrite_encode(new, base, itemsize)
    # only 16 changed elements: record much smaller than full tensor
    assert record.nbytes < new.nbytes

    buf = base.copy()
    dc.overwrite_apply(buf, record, itemsize)
    assert np.array_equal(buf, new)


def test_overwrite_is_idempotent():
    base = _tensor_bytes(_rand((8, 8), seed=7))
    new = _tensor_bytes(_rand((8, 8), seed=8))
    record = dc.overwrite_encode(new, base, 2)
    buf = base.copy()
    dc.overwrite_apply(buf, record, 2)
    dc.overwrite_apply(buf, record, 2)  # re-apply is a no-op
    assert np.array_equal(buf, new)


def test_overwrite_no_change_is_tiny():
    base = _tensor_bytes(_rand((64, 64), seed=9))
    record = dc.overwrite_encode(base.copy(), base, 2)
    assert record.nbytes == 4  # just the u4 count


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def test_checksum_adler32():
    data = _tensor_bytes(_rand((8, 8), seed=10))
    c1 = dc.checksum(data, "adler32")
    c2 = dc.checksum(data.copy(), "adler32")
    assert c1 == c2
    data[0] ^= 0xFF
    assert dc.checksum(data, "adler32") != c1


def test_checksum_unknown_algo():
    with pytest.raises(ValueError, match="Unknown checksum algo"):
        dc.checksum(b"abc", "md5")


# ---------------------------------------------------------------------------
# DeltaWriter / apply_delta round trips
# ---------------------------------------------------------------------------


def _make_state(seed=0, dtype=torch.bfloat16):
    names = ["model.embed.weight", "model.layers.0.mlp.weight", "model.norm.weight"]
    shapes = [(32, 16), (64, 32), (16,)]
    tensors = {n: _rand(s, dtype, seed=seed + i) for i, (n, s) in enumerate(zip(names, shapes))}
    return names, shapes, tensors


@pytest.mark.parametrize("encoding", [dc.ENCODING_XOR, dc.ENCODING_OVERWRITE])
def test_writer_and_state_apply(tmp_path, encoding):
    disk_dir = str(tmp_path / "deltas")
    names, shapes, base_tensors = _make_state(seed=0)
    new_tensors = {n: (t + 0.01).to(t.dtype) for n, t in base_tensors.items()}

    base_state = {n: _tensor_bytes(t).copy() for n, t in base_tensors.items()}

    writer = dc.DeltaWriter(disk_dir, version=1, encoding=encoding, checksum_algo="adler32", model_dtype="bfloat16")
    for n, s in zip(names, shapes):
        writer.add_tensor(n, "bfloat16", list(s), _tensor_bytes(new_tensors[n]), base_state[n])
    vdir = writer.finalize()

    assert dc.list_versions(disk_dir) == [1]
    manifest = dc.read_manifest(vdir)
    assert manifest.version == 1 and len(manifest.tensors) == 3

    # replay into a state dict seeded at base
    dc.apply_delta_to_state(vdir, base_state)
    for n in names:
        assert np.array_equal(base_state[n], _tensor_bytes(new_tensors[n])), n


def test_writer_full_record_for_missing_base(tmp_path):
    disk_dir = str(tmp_path / "deltas")
    t = _rand((8, 4), seed=42)
    writer = dc.DeltaWriter(disk_dir, version=1, encoding="xor", checksum_algo="adler32", model_dtype="bfloat16")
    writer.add_tensor("lm_head.weight", "bfloat16", [8, 4], _tensor_bytes(t), None)
    vdir = writer.finalize()

    state = {}
    dc.apply_delta_to_state(vdir, state)
    assert np.array_equal(state["lm_head.weight"], _tensor_bytes(t))


def test_apply_checksum_mismatch_raises(tmp_path):
    disk_dir = str(tmp_path / "deltas")
    base = _tensor_bytes(_rand((8, 8), seed=1)).copy()
    new = _tensor_bytes(_rand((8, 8), seed=2))
    writer = dc.DeltaWriter(disk_dir, version=1, encoding="xor", checksum_algo="adler32", model_dtype="bfloat16")
    writer.add_tensor("w", "bfloat16", [8, 8], new, base)
    vdir = writer.finalize()

    corrupted = base.copy()
    corrupted[0] ^= 0xFF  # receiver's base diverged from sender's
    with pytest.raises(ValueError, match="Checksum mismatch"):
        dc.apply_delta_to_state(vdir, {"w": corrupted})


def test_writer_atomic_publish(tmp_path):
    disk_dir = str(tmp_path / "deltas")
    writer = dc.DeltaWriter(disk_dir, version=1, encoding="xor", checksum_algo="adler32", model_dtype="bfloat16")
    # not finalized -> not listed
    assert dc.list_versions(disk_dir) == []
    writer.abort()
    assert dc.list_versions(disk_dir) == []


def test_writer_rejects_duplicate_version(tmp_path):
    disk_dir = str(tmp_path / "deltas")
    w = dc.DeltaWriter(disk_dir, version=1, encoding="xor", checksum_algo="adler32", model_dtype="bfloat16")
    w.finalize()
    with pytest.raises(FileExistsError):
        dc.DeltaWriter(disk_dir, version=1, encoding="xor", checksum_algo="adler32", model_dtype="bfloat16")


# ---------------------------------------------------------------------------
# Safetensors patching
# ---------------------------------------------------------------------------


def _write_checkpoint(ckpt_dir, tensors, dtype="bfloat16"):
    dc.write_safetensors_file(
        str(ckpt_dir / "model.safetensors"),
        {n: (dtype, list(t.shape), _tensor_bytes(t).tobytes()) for n, t in tensors.items()},
    )


def test_tensor_locations_roundtrip(tmp_path):
    _, _, tensors = _make_state(seed=0)
    _write_checkpoint(tmp_path, tensors)
    locs = dc.tensor_locations(str(tmp_path))
    assert set(locs) == set(tensors)
    for n, t in tensors.items():
        assert locs[n].shape == list(t.shape)
        assert locs[n].dtype == "bfloat16"
        assert np.array_equal(dc.read_tensor_bytes(locs[n]), _tensor_bytes(t))


@pytest.mark.parametrize("encoding", [dc.ENCODING_XOR, dc.ENCODING_OVERWRITE])
def test_checkpoint_patcher_applies_versions(tmp_path, encoding):
    """End-to-end: base checkpoint on disk + 2 delta versions -> patched bytes."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    disk_dir = str(tmp_path / "deltas")

    names, shapes, v0 = _make_state(seed=0)
    _write_checkpoint(ckpt_dir, v0)

    # sender-side: two successive versions, advancing the base each time
    base_state = {n: _tensor_bytes(t).copy() for n, t in v0.items()}
    versions = {}
    current = v0
    for version in (1, 2):
        nxt = {n: (t + 0.5 * version).to(t.dtype) for n, t in current.items()}
        writer = dc.DeltaWriter(disk_dir, version, encoding, "adler32", "bfloat16")
        for n, s in zip(names, shapes):
            new_bytes = _tensor_bytes(nxt[n])
            writer.add_tensor(n, "bfloat16", list(s), new_bytes, base_state[n])
            base_state[n] = new_bytes.copy()
        writer.finalize()
        versions[version] = nxt
        current = nxt

    # receiver-side: patch the local checkpoint through both versions
    patcher = dc.CheckpointPatcher(str(ckpt_dir))
    for version in (1, 2):
        patcher.apply_version(dc.version_dir(disk_dir, version))
    patcher.close()

    locs = dc.tensor_locations(str(ckpt_dir))
    for n in names:
        assert np.array_equal(dc.read_tensor_bytes(locs[n]), _tensor_bytes(versions[2][n])), n


def test_checkpoint_patcher_full_record_sidecar(tmp_path):
    """A 'full' record for a tensor absent from the checkpoint lands in the sidecar."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    disk_dir = str(tmp_path / "deltas")

    _, _, v0 = _make_state(seed=0)
    _write_checkpoint(ckpt_dir, v0)

    extra = _rand((8, 4), seed=99)
    writer = dc.DeltaWriter(disk_dir, 1, "xor", "adler32", "bfloat16")
    writer.add_tensor("lm_head.weight", "bfloat16", [8, 4], _tensor_bytes(extra), None)
    writer.finalize()

    patcher = dc.CheckpointPatcher(str(ckpt_dir))
    patcher.apply_version(dc.version_dir(disk_dir, 1))

    # sidecar now holds it, and the patcher can address it for later deltas
    assert "lm_head.weight" in patcher.locations
    assert np.array_equal(dc.read_tensor_bytes(patcher.locations["lm_head.weight"]), _tensor_bytes(extra))

    # a second version can xor-patch the sidecar tensor
    extra2 = (extra + 1.0).to(extra.dtype)
    writer = dc.DeltaWriter(disk_dir, 2, "xor", "adler32", "bfloat16")
    writer.add_tensor("lm_head.weight", "bfloat16", [8, 4], _tensor_bytes(extra2), _tensor_bytes(extra).copy())
    writer.finalize()
    patcher.apply_version(dc.version_dir(disk_dir, 2))
    patcher.close()

    locs = dc.tensor_locations(str(ckpt_dir))
    assert np.array_equal(dc.read_tensor_bytes(locs["lm_head.weight"]), _tensor_bytes(extra2))


def test_local_version_marker(tmp_path):
    d = str(tmp_path)
    assert dc.read_local_version(d) == -1
    dc.write_local_version(d, 3)
    assert dc.read_local_version(d) == 3


# ---------------------------------------------------------------------------
# Checkpoint materialization + state loading agree byte-for-byte
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src_dtype,target_dtype", [("bfloat16", "bfloat16"), ("float32", "bfloat16")])
def test_materialize_and_state_agree(tmp_path, src_dtype, target_dtype):
    """Sender base (load_checkpoint_state) must be byte-identical to the
    receiver's materialized local checkpoint — the core sync invariant."""
    src_ckpt = tmp_path / "src"
    src_ckpt.mkdir()
    torch_src = getattr(torch, src_dtype)
    _, _, tensors = _make_state(seed=0, dtype=torch_src)
    _write_checkpoint(src_ckpt, tensors, dtype=src_dtype)

    dst_ckpt = str(tmp_path / "local")
    dc.materialize_local_checkpoint(str(src_ckpt), dst_ckpt, target_dtype)
    sender_state = dc.load_checkpoint_state(str(src_ckpt), target_dtype)

    locs = dc.tensor_locations(dst_ckpt)
    assert set(locs) == set(sender_state)
    for n, loc in locs.items():
        assert loc.dtype == target_dtype
        assert np.array_equal(dc.read_tensor_bytes(loc), sender_state[n]), n
