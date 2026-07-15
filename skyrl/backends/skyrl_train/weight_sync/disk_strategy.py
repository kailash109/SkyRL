"""Disk-based delta weight transfer strategy.

Ships only the bytes that changed between two weight syncs over a shared
filesystem, instead of moving full weights over NCCL. Designed for
non-colocated setups where training and inference are network-separated
(different clusters / datacenters) or where a shared FS is the natural
transport. Follows the design of slime's delta weight sync
(https://github.com/THUDM/slime, docs/en/advanced/delta-weight-sync.md).

Sender side (this module):
- Seeds a CPU base snapshot from the HF checkpoint (``model_path``) — the
  same checkpoint each inference host materializes its local copy from, so
  no full checkpoint ever crosses the shared filesystem.
- Each sync: diffs every gathered HF tensor against the snapshot, encodes
  (xor/overwrite) + zstd-compresses the change into ``disk_dir/v{N}/``, and
  advances the snapshot. On trainer restart the snapshot is rebuilt by
  replaying the published versions.

Receiver side: ``DiskWeightTransferEngine`` (a custom vLLM
``WeightTransferEngine``) applies deltas into a host-local checkpoint and
reloads the patched tensors through ``model.load_weights`` — see
``skyrl/backends/skyrl_train/inference_servers/disk_transfer_engine.py``.
"""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config import InferenceEngineConfig

import numpy as np
import torch
from loguru import logger

from skyrl.backends.skyrl_train.weight_sync import delta_codec
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk, WeightUpdateRequest
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferSender,
    WeightTransferStrategy,
)


@dataclass
class DiskInitInfo(WeightSyncInitInfo):
    """Initialization info for disk-based delta weight transfer."""

    disk_dir: str
    """Shared filesystem directory the trainer publishes deltas to and the
    inference hosts read from."""
    encoding: str = delta_codec.ENCODING_XOR
    checksum_algo: str = "adler32"
    model_dtype_str: str = "bfloat16"
    local_checkpoint_dir: str = ""
    """Host-local directory for the inference side's full checkpoint copy
    (e.g. NVMe). Empty -> the receiver derives a default under tempdir."""
    model_path: str = ""
    """HF checkpoint that seeds the delta base. Defaulted to the policy model
    path by the training worker; the receiver falls back to its own vLLM
    model path when empty."""
    pre_read_hook: str = ""
    """Optional ``"module:function"`` the receiver calls (no args) before
    reading the shared delta directory. Needed for object-store-backed mounts
    whose writes aren't immediately visible across hosts (e.g. a Modal Volume
    needing ``reload()``); POSIX shared filesystems don't need it. Mirrors
    slime's ``--sglang-custom-pull-weights-pre-read-hook``."""

    def for_servers(self, world_size_per_server: int, num_servers: int, dp_size: int = 1) -> List["DiskInitInfo"]:
        """Disk init carries no per-server state; return identical copies."""
        return [copy.deepcopy(self) for _ in range(num_servers)]

    def to_api_payload(self) -> Dict[str, Any]:
        """Payload for /init_weight_transfer_engine.

        Keys must match ``DiskWeightTransferInitInfo`` fields on the vLLM
        engine side (``parse_init_info`` constructs it via ``**payload``).
        """
        return {
            "disk_dir": self.disk_dir,
            "encoding": self.encoding,
            "checksum_algo": self.checksum_algo,
            "model_dtype": self.model_dtype_str,
            "local_checkpoint_dir": self.local_checkpoint_dir,
            "model_path": self.model_path,
            "pre_read_hook": self.pre_read_hook,
        }


@dataclass
class DiskWeightUpdateRequest(WeightUpdateRequest):
    """Request for disk-based weight transfer: metadata + target version."""

    version: int = 0


class DiskWeightTransferSender(WeightTransferSender):
    """Publishes per-sync weight deltas to a shared filesystem.

    Rank 0 holds a CPU base snapshot (``{name: uint8 array}``) of the last
    published weights: one full-model CPU copy. Non-rank-0 workers only drain
    the chunk iterator (weight extraction uses collective ops).
    """

    def __init__(
        self,
        init_info: DiskInitInfo,
        inference_client: "RemoteInferenceClient",
    ) -> None:
        self._init_info = init_info
        self._inference_client = inference_client
        self._base: Optional[Dict[str, np.ndarray]] = None
        self._next_version: Optional[int] = None
        # torch dtype name, e.g. "bfloat16"
        self._dtype_name = init_info.model_dtype_str.split(".")[-1]

    def _ensure_base(self) -> None:
        """Seed (or rebuild) the base snapshot on rank 0.

        Seeds from the HF checkpoint, then replays any versions already
        published under ``disk_dir`` — so a restarted trainer resumes with a
        base identical to what the inference hosts hold. Requires a fresh
        ``disk_dir`` per run (stale versions from another run will corrupt
        the delta chain).
        """
        if self._base is not None:
            return
        if not self._init_info.model_path:
            raise ValueError("DiskInitInfo.model_path must be set to seed the delta base snapshot")
        logger.info(
            f"Seeding disk weight-sync base snapshot from {self._init_info.model_path} " f"(dtype={self._dtype_name})"
        )
        state = delta_codec.load_checkpoint_state(self._init_info.model_path, self._dtype_name)
        versions = delta_codec.list_versions(self._init_info.disk_dir)
        for v in versions:
            logger.info(f"Replaying published delta v{v} into base snapshot (trainer resume)")
            delta_codec.apply_delta_to_state(delta_codec.version_dir(self._init_info.disk_dir, v), state)
        self._base = state
        self._next_version = (versions[-1] + 1) if versions else 1

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
    ) -> None:
        """Diff chunks against the base snapshot, publish one delta version,
        then tell the inference engines to apply it.

        Args:
            chunks: Iterable of WeightChunk objects to send.
            weight_metadata: Unused (metadata is collected from the chunks,
                which the sender must fully iterate anyway before publishing).
                Kept for interface compatibility.
        """
        dist_initialized = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if dist_initialized else 0

        if rank != 0:
            # Extraction may use collective ops; every rank must drain.
            for _ in chunks:
                pass
            torch.distributed.barrier()
            return

        self._ensure_base()
        assert self._base is not None and self._next_version is not None
        version = self._next_version

        names: List[str] = []
        dtype_names: List[str] = []
        shapes: List[List[int]] = []

        writer = delta_codec.DeltaWriter(
            self._init_info.disk_dir,
            version,
            encoding=self._init_info.encoding,
            checksum_algo=self._init_info.checksum_algo,
            model_dtype=self._dtype_name,
        )
        try:
            for chunk in chunks:
                for name, tensor, shape, dtype_str in zip(chunk.names, chunk.tensors, chunk.shapes, chunk.dtypes):
                    dtype_name = dtype_str.split(".")[-1]
                    if dtype_name != self._dtype_name:
                        raise ValueError(
                            f"Disk delta sync requires a single model dtype: tensor {name!r} is "
                            f"{dtype_name}, expected {self._dtype_name} (inference_engine.model_dtype)"
                        )
                    new_bytes = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().reshape(-1)
                    base_bytes = self._base.get(name)
                    if base_bytes is not None and base_bytes.nbytes != new_bytes.nbytes:
                        raise ValueError(
                            f"Byte-size mismatch for {name!r}: base has {base_bytes.nbytes}, "
                            f"new tensor has {new_bytes.nbytes}. The checkpoint at "
                            f"{self._init_info.model_path} does not match the trained model."
                        )
                    if base_bytes is None:
                        logger.warning(
                            f"Tensor {name!r} absent from checkpoint base; publishing full bytes "
                            f"(expected for tied/aliased params on the first occurrence)"
                        )
                    writer.add_tensor(name, dtype_name, list(shape), new_bytes, base_bytes)
                    self._base[name] = new_bytes  # advance the snapshot (lossless)
                    names.append(name)
                    dtype_names.append(dtype_name)
                    shapes.append(list(shape))
        except Exception:
            writer.abort()
            raise
        vdir = writer.finalize()
        self._next_version = version + 1
        logger.info(f"Published weight delta v{version} ({len(names)} tensors) to {vdir}")

        # Drive the receive through SkyRL's chunked lifecycle so the load is
        # wrapped with set_current_vllm_config (matches NCCL/IPC paths).
        await self._inference_client.start_weight_update(is_checkpoint_format=True)
        update_info: Dict[str, Any] = {
            "version": version,
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
        }
        await self._inference_client.update_weights_disk(update_info)
        await self._inference_client.finish_weight_update()

        if dist_initialized:
            torch.distributed.barrier()

    def teardown(self) -> None:
        """Drop the base snapshot (no process groups to clean up)."""
        self._base = None


class DiskTransferStrategy(WeightTransferStrategy):
    """Factory for disk-based delta weight transfer.

    All methods are static — no instance state needed.
    """

    @staticmethod
    def create_init_info(ie_cfg: "InferenceEngineConfig", inference_world_size: Optional[int] = None) -> DiskInitInfo:
        """Create init info from config. ``inference_world_size`` is unused."""
        if not ie_cfg.weight_sync_disk_dir:
            raise ValueError(
                "generator.inference_engine.weight_sync_disk_dir must be set when "
                "weight_sync_backend='disk' (shared filesystem path visible to both "
                "the trainer and the inference servers)"
            )
        return DiskInitInfo(
            disk_dir=ie_cfg.weight_sync_disk_dir,
            encoding=ie_cfg.weight_sync_delta_encoding,
            checksum_algo=ie_cfg.weight_sync_delta_checksum,
            model_dtype_str=ie_cfg.model_dtype,
            local_checkpoint_dir=ie_cfg.weight_sync_local_ckpt_dir or "",
            pre_read_hook=ie_cfg.weight_sync_disk_pre_read_hook or "",
            override_existing_receiver=not ie_cfg.run_engines_locally,
        )

    @staticmethod
    def create_sender(
        init_info: DiskInitInfo,
        inference_client: "RemoteInferenceClient",
    ) -> DiskWeightTransferSender:
        return DiskWeightTransferSender(init_info=init_info, inference_client=inference_client)

    @staticmethod
    def get_vllm_transfer_engine() -> type:
        """Return the receive-side engine class (SkyRL's custom disk engine)."""
        from skyrl.backends.skyrl_train.inference_servers.disk_transfer_engine import (
            DiskWeightTransferEngine,
        )

        return DiskWeightTransferEngine
