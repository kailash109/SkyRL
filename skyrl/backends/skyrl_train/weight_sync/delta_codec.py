"""Lossless byte-level delta codec for disk-based weight sync.

Implements the encode/apply primitives for shipping *changed bytes* between
two weight versions over a shared filesystem, following the design of
slime's disk delta sync (THUDM/slime, ``slime/utils/disk_delta.py``):

- ``xor`` encoding: ``new ^ old`` over uint8 views. Unchanged bytes become
  zeros, which zstd crushes. Apply is the same XOR (involution).
- ``overwrite`` encoding: ``[count:u4][positions:u4...][raw new-value bytes]``
  at element granularity. Larger on the wire but idempotent (safe to
  re-apply after a crashed apply).
- All records are zstd-compressed (level 1) and carry a checksum of the
  *post-apply* bytes so the receiver can verify integrity.

This module is CPU-only (numpy + zstandard + stdlib); torch is imported
lazily only inside checkpoint-materialization helpers. It must stay
importable without vLLM so CPU tests can exercise it.

On-disk layout (shared ``disk_dir``)::

    disk_dir/
      v1/
        manifest.json   # DeltaManifest: per-tensor metadata + offsets
        delta.bin       # concatenated zstd frames, one per tensor
      v2/
        ...

Version directories are written to ``v{N}.tmp`` and atomically renamed so a
partially-written version is never visible to readers.
"""

import json
import mmap
import os
import shutil
import struct
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import zstandard

DELTA_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
DELTA_BIN_NAME = "delta.bin"
LOCAL_VERSION_MARKER = ".skyrl_weight_version"
SIDECAR_NAME = "skyrl_extra.safetensors"

ENCODING_XOR = "xor"
ENCODING_OVERWRITE = "overwrite"
# Per-record fallback when the sender has no base for a tensor (e.g. a tied
# lm_head that is absent from the HF checkpoint): raw new bytes, zstd'd.
ENCODING_FULL = "full"

_ZSTD_LEVEL = 1

# torch dtype string ("bfloat16") -> element size in bytes
TORCH_DTYPE_NBYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}

# safetensors dtype tag <-> torch dtype string
SAFETENSORS_TO_TORCH_DTYPE = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}
TORCH_TO_SAFETENSORS_DTYPE = {v: k for k, v in SAFETENSORS_TO_TORCH_DTYPE.items()}


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def checksum(data, algo: str) -> str:
    """Checksum bytes-like data. ``algo`` is ``"adler32"`` or ``"xxh3-128"``.

    ``adler32`` (stdlib) is the no-extra-dependency default; ``xxh3-128``
    requires the ``xxhash`` package and is faster/stronger for large tensors.
    """
    if isinstance(data, np.ndarray):
        data = memoryview(data).cast("B")
    if algo == "adler32":
        return format(zlib.adler32(data) & 0xFFFFFFFF, "08x")
    elif algo == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128_hexdigest(data)
    raise ValueError(f"Unknown checksum algo: {algo}. Supported: adler32, xxh3-128")


# ---------------------------------------------------------------------------
# Encodings (operate on uint8 numpy arrays)
# ---------------------------------------------------------------------------


def xor_encode(new: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Byte-level ``new ^ base``. Involution: apply == encode."""
    if new.nbytes != base.nbytes:
        raise ValueError(f"Byte-length mismatch: new={new.nbytes} base={base.nbytes}")
    return np.bitwise_xor(new.view(np.uint8), base.view(np.uint8))


def xor_apply(base: np.ndarray, delta: np.ndarray) -> None:
    """Apply an XOR delta into ``base`` in place (``base ^= delta``)."""
    if base.nbytes != delta.nbytes:
        raise ValueError(f"Byte-length mismatch: base={base.nbytes} delta={delta.nbytes}")
    np.bitwise_xor(base.view(np.uint8), delta.view(np.uint8), out=base.view(np.uint8))


def overwrite_encode(new: np.ndarray, base: np.ndarray, itemsize: int) -> np.ndarray:
    """Encode changed elements as ``[count:u4][positions:u4...][new bytes]``.

    Positions are element indices (``itemsize``-byte units), matching slime's
    ``overwrite_encode``. Idempotent to apply, unlike xor.
    """
    new_u8 = new.view(np.uint8).reshape(-1, itemsize)
    base_u8 = base.view(np.uint8).reshape(-1, itemsize)
    if new_u8.shape != base_u8.shape:
        raise ValueError(f"Shape mismatch: new={new_u8.shape} base={base_u8.shape}")
    if new_u8.shape[0] > 0xFFFFFFFF:
        raise ValueError(f"Tensor has {new_u8.shape[0]} elements; overwrite positions are u32")
    changed = (new_u8 != base_u8).any(axis=1)
    pos = np.flatnonzero(changed).astype("<u4")
    return np.concatenate(
        [
            np.array([pos.size], dtype="<u4").view(np.uint8),
            pos.view(np.uint8),
            new_u8[changed].reshape(-1),
        ]
    )


def overwrite_apply(base: np.ndarray, record: np.ndarray, itemsize: int) -> None:
    """Scatter an overwrite record into ``base`` (uint8 view) in place."""
    record = record.view(np.uint8)
    count = int(record[:4].view("<u4")[0])
    pos_end = 4 + 4 * count
    pos = record[4:pos_end].view("<u4")
    values = record[pos_end : pos_end + count * itemsize].reshape(count, itemsize)
    base.view(np.uint8).reshape(-1, itemsize)[pos] = values


# ---------------------------------------------------------------------------
# Manifest / delta file
# ---------------------------------------------------------------------------


@dataclass
class TensorDeltaMeta:
    """Per-tensor record metadata within one version's delta.bin."""

    name: str
    dtype: str  # torch dtype string, e.g. "bfloat16"
    shape: List[int]
    encoding: str  # "xor" | "overwrite" | "full"
    offset: int  # byte offset of the zstd frame in delta.bin
    compressed_nbytes: int
    raw_nbytes: int  # decompressed record size
    post_apply_checksum: str  # checksum of the tensor's raw bytes AFTER apply


@dataclass
class DeltaManifest:
    format_version: int
    version: int
    encoding: str
    checksum_algo: str
    model_dtype: str
    tensors: List[TensorDeltaMeta] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "DeltaManifest":
        data = json.loads(text)
        data["tensors"] = [TensorDeltaMeta(**t) for t in data["tensors"]]
        return cls(**data)


def version_dir(disk_dir: str, version: int) -> str:
    return os.path.join(disk_dir, f"v{version}")


def list_versions(disk_dir: str) -> List[int]:
    """Sorted list of finalized version numbers under ``disk_dir``."""
    if not os.path.isdir(disk_dir):
        return []
    versions = []
    for entry in os.listdir(disk_dir):
        if entry.startswith("v") and not entry.endswith(".tmp"):
            try:
                v = int(entry[1:])
            except ValueError:
                continue
            if os.path.isfile(os.path.join(disk_dir, entry, MANIFEST_NAME)):
                versions.append(v)
    return sorted(versions)


class DeltaWriter:
    """Writes one version's delta records + manifest, with atomic publish."""

    def __init__(self, disk_dir: str, version: int, encoding: str, checksum_algo: str, model_dtype: str):
        if encoding not in (ENCODING_XOR, ENCODING_OVERWRITE):
            raise ValueError(f"Unknown delta encoding: {encoding}")
        self._final_dir = version_dir(disk_dir, version)
        self._tmp_dir = self._final_dir + ".tmp"
        if os.path.exists(self._final_dir):
            raise FileExistsError(f"Delta version already published: {self._final_dir}")
        shutil.rmtree(self._tmp_dir, ignore_errors=True)  # stale crash leftovers
        os.makedirs(self._tmp_dir)
        self._encoding = encoding
        self._checksum_algo = checksum_algo
        self._manifest = DeltaManifest(
            format_version=DELTA_FORMAT_VERSION,
            version=version,
            encoding=encoding,
            checksum_algo=checksum_algo,
            model_dtype=model_dtype,
        )
        self._bin = open(os.path.join(self._tmp_dir, DELTA_BIN_NAME), "wb")
        self._offset = 0
        self._compressor = zstandard.ZstdCompressor(level=_ZSTD_LEVEL)

    def add_tensor(
        self,
        name: str,
        dtype: str,
        shape: List[int],
        new_bytes: np.ndarray,
        base_bytes: Optional[np.ndarray],
    ) -> None:
        """Encode ``new_bytes`` (uint8 view of the tensor) against ``base_bytes``.

        ``base_bytes=None`` emits a ``full`` record (raw new bytes) for tensors
        the sender has no base for.
        """
        new_u8 = new_bytes.view(np.uint8).reshape(-1)
        if base_bytes is None:
            encoding = ENCODING_FULL
            record = new_u8
        elif self._encoding == ENCODING_XOR:
            encoding = ENCODING_XOR
            record = xor_encode(new_u8, base_bytes)
        else:
            encoding = ENCODING_OVERWRITE
            record = overwrite_encode(new_u8, base_bytes, TORCH_DTYPE_NBYTES[dtype])
        frame = self._compressor.compress(record.tobytes())
        self._bin.write(frame)
        self._manifest.tensors.append(
            TensorDeltaMeta(
                name=name,
                dtype=dtype,
                shape=list(shape),
                encoding=encoding,
                offset=self._offset,
                compressed_nbytes=len(frame),
                raw_nbytes=record.nbytes,
                post_apply_checksum=checksum(new_u8, self._checksum_algo),
            )
        )
        self._offset += len(frame)

    def finalize(self) -> str:
        """Write the manifest and atomically publish the version directory."""
        self._bin.close()
        with open(os.path.join(self._tmp_dir, MANIFEST_NAME), "w") as f:
            f.write(self._manifest.to_json())
        os.rename(self._tmp_dir, self._final_dir)
        return self._final_dir

    def abort(self) -> None:
        self._bin.close()
        shutil.rmtree(self._tmp_dir, ignore_errors=True)


def read_manifest(vdir: str) -> DeltaManifest:
    with open(os.path.join(vdir, MANIFEST_NAME)) as f:
        return DeltaManifest.from_json(f.read())


def iter_delta_records(
    vdir: str, manifest: Optional[DeltaManifest] = None
) -> Iterator[Tuple[TensorDeltaMeta, np.ndarray]]:
    """Yield ``(meta, raw_record_bytes)`` per tensor, decompressing frames."""
    if manifest is None:
        manifest = read_manifest(vdir)
    decompressor = zstandard.ZstdDecompressor()
    with open(os.path.join(vdir, DELTA_BIN_NAME), "rb") as f:
        for meta in manifest.tensors:
            f.seek(meta.offset)
            frame = f.read(meta.compressed_nbytes)
            raw = decompressor.decompress(frame, max_output_size=meta.raw_nbytes)
            yield meta, np.frombuffer(raw, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Generic delta apply
# ---------------------------------------------------------------------------


def try_read_complete_manifest(vdir: str) -> Optional[DeltaManifest]:
    """Read a version's manifest, returning None unless the version is fully
    readable: manifest parses AND delta.bin has all the bytes the manifest
    references. Guards against eventually-consistent shared mounts where a
    file entry appears before its content has synced."""
    try:
        manifest = read_manifest(vdir)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    expected_nbytes = sum(t.compressed_nbytes for t in manifest.tensors)
    try:
        actual_nbytes = os.path.getsize(os.path.join(vdir, DELTA_BIN_NAME))
    except OSError:
        return None
    if actual_nbytes < expected_nbytes:
        return None
    return manifest


def apply_delta(
    vdir: str,
    get_buffer: Callable[[TensorDeltaMeta], Optional[np.ndarray]],
    set_full: Callable[[TensorDeltaMeta, np.ndarray], None],
    manifest: Optional[DeltaManifest] = None,
) -> DeltaManifest:
    """Apply one version's records via caller-provided buffer accessors.

    ``get_buffer(meta)`` returns a writable uint8 array for the tensor (or
    None if unknown, valid only for ``full`` records). ``set_full(meta, bytes)``
    stores a full-record tensor. Verifies each post-apply checksum and raises
    ``ValueError`` on mismatch. Pass ``manifest`` when it was already read
    (and validated) to avoid re-reading a possibly-inconsistent mount.
    """
    if manifest is None:
        manifest = read_manifest(vdir)
    for meta, record in iter_delta_records(vdir, manifest):
        if meta.encoding == ENCODING_FULL:
            set_full(meta, record)
            continue
        buf = get_buffer(meta)
        if buf is None:
            raise KeyError(f"No target buffer for tensor {meta.name!r} (encoding={meta.encoding})")
        if meta.encoding == ENCODING_XOR:
            xor_apply(buf, record)
        elif meta.encoding == ENCODING_OVERWRITE:
            overwrite_apply(buf, record, TORCH_DTYPE_NBYTES[meta.dtype])
        else:
            raise ValueError(f"Unknown record encoding: {meta.encoding}")
        actual = checksum(buf, manifest.checksum_algo)
        if actual != meta.post_apply_checksum:
            raise ValueError(
                f"Checksum mismatch after applying {vdir} for tensor {meta.name!r}: "
                f"expected {meta.post_apply_checksum}, got {actual}. "
                f"The local checkpoint is likely corrupted or out of sync."
            )
    return manifest


def apply_delta_to_state(vdir: str, state: Dict[str, np.ndarray]) -> DeltaManifest:
    """Apply one version into an in-memory ``{name: uint8 array}`` state dict."""

    def get_buffer(meta: TensorDeltaMeta) -> Optional[np.ndarray]:
        return state.get(meta.name)

    def set_full(meta: TensorDeltaMeta, raw: np.ndarray) -> None:
        state[meta.name] = raw.copy()

    return apply_delta(vdir, get_buffer, set_full)


# ---------------------------------------------------------------------------
# Safetensors: header parsing, location index, in-place patching
# ---------------------------------------------------------------------------


@dataclass
class TensorLocation:
    file_path: str
    offset: int  # absolute byte offset of tensor data within the file
    nbytes: int
    dtype: str  # torch dtype string
    shape: List[int]


def _read_safetensors_header(path: str) -> Tuple[Dict[str, Any], int]:
    """Return (header_dict, data_start_offset) for a safetensors file."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def tensor_locations(ckpt_dir: str) -> Dict[str, TensorLocation]:
    """Map tensor name -> byte location across all safetensors in a directory."""
    locations: Dict[str, TensorLocation] = {}
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith(".safetensors"):
            continue
        path = os.path.join(ckpt_dir, fname)
        header, data_start = _read_safetensors_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            locations[name] = TensorLocation(
                file_path=path,
                offset=data_start + start,
                nbytes=end - start,
                dtype=SAFETENSORS_TO_TORCH_DTYPE[info["dtype"]],
                shape=list(info["shape"]),
            )
    return locations


def write_safetensors_file(path: str, tensors: Dict[str, Tuple[str, List[int], bytes]]) -> None:
    """Write a safetensors file from ``{name: (torch_dtype, shape, raw_bytes)}``.

    Byte-level writer so we never round-trip through frameworks that lack
    bfloat16 (numpy) — used for the sidecar file holding tensors absent from
    the original checkpoint.
    """
    header: Dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": TORCH_TO_SAFETENSORS_DTYPE[dtype],
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    header_bytes = json.dumps(header).encode()
    # Pad header to 8-byte alignment (safetensors convention).
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for _, (_, _, raw) in tensors.items():
            f.write(raw)
    os.replace(tmp, path)


def read_tensor_bytes(loc: TensorLocation) -> np.ndarray:
    with open(loc.file_path, "rb") as f:
        f.seek(loc.offset)
        return np.frombuffer(f.read(loc.nbytes), dtype=np.uint8)


class CheckpointPatcher:
    """Applies delta versions in place into a local safetensors checkpoint.

    Full records land in a sidecar safetensors file (``skyrl_extra.safetensors``)
    so tensors absent from the original checkpoint (e.g. tied lm_head emitted
    by the trainer's extractor) participate in later deltas uniformly.
    """

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self._locations = tensor_locations(ckpt_dir)
        self._mmaps: Dict[str, Tuple[Any, mmap.mmap]] = {}  # file_path -> (fh, mmap)

    def _writable_view(self, loc: TensorLocation) -> np.ndarray:
        if loc.file_path not in self._mmaps:
            fh = open(loc.file_path, "r+b")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_WRITE)
            self._mmaps[loc.file_path] = (fh, mm)
        _, mm = self._mmaps[loc.file_path]
        return np.frombuffer(mm, dtype=np.uint8, count=loc.nbytes, offset=loc.offset)

    def apply_version(self, vdir: str, manifest: Optional[DeltaManifest] = None) -> DeltaManifest:
        pending_full: Dict[str, Tuple[str, List[int], bytes]] = {}

        def get_buffer(meta: TensorDeltaMeta) -> Optional[np.ndarray]:
            loc = self._locations.get(meta.name)
            if loc is None:
                return None
            expected_nbytes = int(np.prod(meta.shape or [1])) * TORCH_DTYPE_NBYTES[meta.dtype]
            if loc.nbytes != expected_nbytes:
                raise ValueError(
                    f"Size mismatch for {meta.name!r}: checkpoint has {loc.nbytes} bytes, "
                    f"delta expects {expected_nbytes}"
                )
            return self._writable_view(loc)

        def set_full(meta: TensorDeltaMeta, raw: np.ndarray) -> None:
            loc = self._locations.get(meta.name)
            if loc is not None:
                # Tensor exists (e.g. re-sent full): overwrite in place.
                self._writable_view(loc)[:] = raw
            else:
                pending_full[meta.name] = (meta.dtype, meta.shape, raw.tobytes())

        manifest = apply_delta(vdir, get_buffer, set_full, manifest=manifest)

        if pending_full:
            self._flush_sidecar(pending_full)
        self.flush()
        return manifest

    def _flush_sidecar(self, new_tensors: Dict[str, Tuple[str, List[int], bytes]]) -> None:
        """Rewrite the sidecar including existing entries + new full tensors."""
        sidecar_path = os.path.join(self.ckpt_dir, SIDECAR_NAME)
        existing: Dict[str, Tuple[str, List[int], bytes]] = {}
        if os.path.exists(sidecar_path):
            # Close any open mmap on the sidecar before rewriting it.
            if sidecar_path in self._mmaps:
                fh, mm = self._mmaps.pop(sidecar_path)
                mm.close()
                fh.close()
            for name, loc in tensor_locations_for_file(sidecar_path).items():
                if name not in new_tensors:
                    existing[name] = (loc.dtype, loc.shape, read_tensor_bytes(loc).tobytes())
        existing.update(new_tensors)
        write_safetensors_file(sidecar_path, existing)
        # Refresh the location index to include sidecar entries.
        self._locations = tensor_locations(self.ckpt_dir)

    def flush(self) -> None:
        for _, mm in self._mmaps.values():
            mm.flush()

    def close(self) -> None:
        for fh, mm in self._mmaps.values():
            mm.close()
            fh.close()
        self._mmaps.clear()

    @property
    def locations(self) -> Dict[str, TensorLocation]:
        return self._locations


def tensor_locations_for_file(path: str) -> Dict[str, TensorLocation]:
    header, data_start = _read_safetensors_header(path)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        out[name] = TensorLocation(
            file_path=path,
            offset=data_start + start,
            nbytes=end - start,
            dtype=SAFETENSORS_TO_TORCH_DTYPE[info["dtype"]],
            shape=list(info["shape"]),
        )
    return out


# ---------------------------------------------------------------------------
# Local version marker
# ---------------------------------------------------------------------------


def read_local_version(ckpt_dir: str) -> int:
    """Version the local checkpoint has been patched to (-1 = not materialized)."""
    path = os.path.join(ckpt_dir, LOCAL_VERSION_MARKER)
    if not os.path.exists(path):
        return -1
    with open(path) as f:
        return int(f.read().strip())


def write_local_version(ckpt_dir: str, version: int) -> None:
    path = os.path.join(ckpt_dir, LOCAL_VERSION_MARKER)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(version))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Checkpoint materialization (lazy torch for dtype conversion)
# ---------------------------------------------------------------------------


def resolve_checkpoint_dir(model_path: str) -> str:
    """Resolve a model path or HF repo id to a local directory of safetensors."""
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, allow_patterns=["*.safetensors", "*.safetensors.index.json"])


def materialize_local_checkpoint(model_path: str, dst_dir: str, target_dtype: str) -> None:
    """Copy (or dtype-convert) a checkpoint's safetensors into ``dst_dir``.

    The result is the byte-identical base both sides diff/patch against:
    the sender seeds its base snapshot with :func:`load_checkpoint_state`
    using the same ``target_dtype`` conversion.
    """
    src_dir = resolve_checkpoint_dir(model_path)
    os.makedirs(dst_dir, exist_ok=True)
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".safetensors"):
            continue
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        header, _ = _read_safetensors_header(src)
        dtypes = {info["dtype"] for name, info in header.items() if name != "__metadata__"}
        if dtypes <= {TORCH_TO_SAFETENSORS_DTYPE[target_dtype]}:
            tmp = dst + ".tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        else:
            converted: Dict[str, Tuple[str, List[int], bytes]] = {}
            for name, loc in tensor_locations_for_file(src).items():
                converted[name] = (
                    target_dtype,
                    loc.shape,
                    _convert_bytes_dtype(read_tensor_bytes(loc), loc.dtype, target_dtype, loc.shape),
                )
            write_safetensors_file(dst, converted)


def load_checkpoint_state(model_path: str, target_dtype: str) -> Dict[str, np.ndarray]:
    """Load a checkpoint as ``{name: writable uint8 array}`` in ``target_dtype``."""
    src_dir = resolve_checkpoint_dir(model_path)
    state: Dict[str, np.ndarray] = {}
    for name, loc in tensor_locations(src_dir).items():
        raw = read_tensor_bytes(loc)
        if loc.dtype != target_dtype:
            raw = np.frombuffer(_convert_bytes_dtype(raw, loc.dtype, target_dtype, loc.shape), dtype=np.uint8)
        state[name] = raw.copy()  # writable
    return state


def _convert_bytes_dtype(raw: np.ndarray, src_dtype: str, dst_dtype: str, shape: List[int]) -> bytes:
    """Deterministically convert raw tensor bytes between torch dtypes."""
    import torch

    src = getattr(torch, src_dtype)
    dst = getattr(torch, dst_dtype)
    t = torch.from_numpy(raw.copy()).view(src).reshape(shape)
    return t.to(dst).contiguous().view(torch.uint8).numpy().tobytes()
