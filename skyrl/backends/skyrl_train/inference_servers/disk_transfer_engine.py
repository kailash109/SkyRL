"""Custom vLLM WeightTransferEngine that receives weight updates from disk.

Receive side of SkyRL's disk-based delta weight sync (see
``skyrl/backends/skyrl_train/weight_sync/disk_strategy.py`` for the sender
and the on-disk format). Registered with vLLM's ``WeightTransferEngineFactory``
under the backend name ``"disk"`` (registration happens at import time of
``new_inference_worker_wrap.py``, which vLLM imports in every worker process
via ``--worker-extension-cls``).

Per weight update:

1. **Per-host apply** (one worker per host, serialized by a file lock): read
   the compressed delta version(s) from the shared ``disk_dir``, patch them
   in place into a host-local full checkpoint copy, verify per-tensor
   checksums. The local copy is materialized from the HF checkpoint on first
   use — the full model never crosses the shared filesystem.
2. **Per-worker load**: read the requested tensors (full, unsharded HF
   format) from the patched local checkpoint and hand them to
   ``model.load_weights`` in bounded-size batches — vLLM re-shards/fuses
   them, so TP/PP/quantized serving works exactly like the NCCL path.

This module must only be imported inside vLLM worker processes (it imports
vLLM at module level).
"""

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, List

import torch
from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

from skyrl.backends.skyrl_train.weight_sync import delta_codec

logger = init_logger(__name__)

# Hand tensors to model.load_weights in batches of at most this many bytes.
_LOAD_BATCH_NBYTES = 2 * 1024**3

# How long to wait for a published delta version to become visible on the
# shared filesystem (covers small propagation lag; POSIX FS needs ~none).
_VERSION_VISIBILITY_TIMEOUT_S = 120.0


@dataclass
class DiskWeightTransferInitInfo(WeightTransferInitInfo):
    """Init info for disk transfer. Field names must match the payload built
    by ``DiskInitInfo.to_api_payload()`` on the sender side."""

    disk_dir: str
    encoding: str = delta_codec.ENCODING_XOR
    checksum_algo: str = "adler32"
    model_dtype: str = "bfloat16"
    local_checkpoint_dir: str = ""
    model_path: str = ""
    pre_read_hook: str = ""
    """Optional "module:function" called (no args) before reading the shared
    delta directory — for mounts that need an explicit refresh to see writes
    from other hosts (e.g. modal.Volume.reload()). Empty -> no-op."""


@dataclass
class DiskWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for disk transfer: tensor metadata + target delta version."""

    names: List[str] = field(default_factory=list)
    dtype_names: List[str] = field(default_factory=list)
    shapes: List[List[int]] = field(default_factory=list)
    version: int = 0
    model_path: str = ""
    """Fallback checkpoint path, injected by ``NewInferenceWorkerWrap`` from
    the worker's own vLLM model config when the init info didn't carry one."""


class DiskWeightTransferEngine(WeightTransferEngine[DiskWeightTransferInitInfo, DiskWeightTransferUpdateInfo]):
    """Applies published weight deltas from a shared filesystem into a
    host-local checkpoint, then loads the patched tensors into the model."""

    init_info_cls = DiskWeightTransferInitInfo
    update_info_cls = DiskWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        parallel_config: ParallelConfig,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, parallel_config, model)
        self._init_info: DiskWeightTransferInitInfo | None = None
        self._local_dir: str | None = None
        self._pre_read_hook_fn: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # WeightTransferEngine interface
    # ------------------------------------------------------------------

    def init_transfer_engine(self, init_info: DiskWeightTransferInitInfo) -> None:
        """Store config. The local checkpoint is materialized lazily on the
        first ``receive_weights`` (which knows the model path for sure)."""
        if not init_info.disk_dir:
            raise ValueError("DiskWeightTransferInitInfo.disk_dir must be set")
        self._init_info = init_info
        logger.info(
            "Disk weight transfer engine initialized: disk_dir=%s encoding=%s",
            init_info.disk_dir,
            init_info.encoding,
        )

    def receive_weights(
        self,
        update_info: DiskWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self._init_info is None:
            raise RuntimeError(
                "init_transfer_engine must be called before receive_weights " "(via /init_weight_transfer_engine)"
            )
        model_path = self._init_info.model_path or update_info.model_path
        if not model_path:
            raise ValueError(
                "No checkpoint path available to materialize the local base: "
                "set model_path in the init info or update info"
            )

        local_dir = self._resolve_local_dir(model_path)
        self._apply_versions_locked(local_dir, model_path, update_info.version)
        self._load_from_local(local_dir, update_info, load_weights)

    def shutdown(self) -> None:
        pass

    @staticmethod
    def trainer_send_weights(iterator, trainer_args) -> None:
        raise NotImplementedError(
            "Disk transfer does not send weights through vLLM; the trainer side "
            "is skyrl's DiskWeightTransferSender (weight_sync/disk_strategy.py)"
        )

    # ------------------------------------------------------------------
    # Host-local checkpoint management
    # ------------------------------------------------------------------

    def _resolve_local_dir(self, model_path: str) -> str:
        if self._local_dir is not None:
            return self._local_dir
        local_dir = self._init_info.local_checkpoint_dir
        if not local_dir:
            # Derive a stable per-(model, disk_dir) default so all workers on
            # this host share one copy across restarts.
            import hashlib

            digest = hashlib.sha256(f"{model_path}|{self._init_info.disk_dir}".encode()).hexdigest()[:16]
            local_dir = os.path.join(tempfile.gettempdir(), "skyrl_disk_weight_sync", digest)
        os.makedirs(local_dir, exist_ok=True)
        self._local_dir = local_dir
        return local_dir

    def _apply_versions_locked(self, local_dir: str, model_path: str, target_version: int) -> None:
        """Materialize the local base and apply pending versions, once per host.

        All workers on a host funnel through a file lock; the first one in
        does the work, later ones see the bumped version marker and skip.
        """
        from filelock import FileLock

        with FileLock(os.path.join(local_dir, ".skyrl_apply.lock")):
            if delta_codec.read_local_version(local_dir) < 0:
                logger.info(
                    "Materializing local checkpoint from %s into %s (dtype=%s)",
                    model_path,
                    local_dir,
                    self._init_info.model_dtype,
                )
                start = time.perf_counter()
                delta_codec.materialize_local_checkpoint(model_path, local_dir, self._init_info.model_dtype)
                delta_codec.write_local_version(local_dir, 0)
                logger.info("Local checkpoint materialized in %.1fs", time.perf_counter() - start)

            current = delta_codec.read_local_version(local_dir)
            if current >= target_version:
                return  # another worker on this host already applied

            patcher = delta_codec.CheckpointPatcher(local_dir)
            try:
                for version in range(current + 1, target_version + 1):
                    vdir = delta_codec.version_dir(self._init_info.disk_dir, version)
                    manifest = self._wait_for_version(vdir, version)
                    start = time.perf_counter()
                    patcher.apply_version(vdir, manifest=manifest)
                    delta_codec.write_local_version(local_dir, version)
                    logger.info(
                        "Applied weight delta v%d into %s in %.1fs",
                        version,
                        local_dir,
                        time.perf_counter() - start,
                    )
            finally:
                patcher.close()

    def _run_pre_read_hook(self) -> None:
        """Refresh the shared mount's view, when configured (e.g. Volume.reload())."""
        if not self._init_info.pre_read_hook:
            return
        if self._pre_read_hook_fn is None:
            import importlib

            module_name, _, func_name = self._init_info.pre_read_hook.partition(":")
            if not module_name or not func_name:
                raise ValueError(f"pre_read_hook must be 'module:function', got {self._init_info.pre_read_hook!r}")
            self._pre_read_hook_fn = getattr(importlib.import_module(module_name), func_name)
        self._pre_read_hook_fn()

    def _wait_for_version(self, vdir: str, version: int) -> "delta_codec.DeltaManifest":
        """Poll until a published version is *fully readable*, refreshing the
        mount each attempt. Mere existence is not enough on eventually-
        consistent mounts (a file entry can appear before its content has
        synced), so this validates that the manifest parses and delta.bin
        carries all referenced bytes, and returns the validated manifest.

        Only ever runs while holding the per-host apply lock, so hook-driven
        mount refreshes can't race another worker's in-flight delta read.
        """
        deadline = time.monotonic() + _VERSION_VISIBILITY_TIMEOUT_S
        while True:
            self._run_pre_read_hook()
            manifest = delta_codec.try_read_complete_manifest(vdir)
            if manifest is not None:
                return manifest
            if time.monotonic() > deadline:
                raise FileNotFoundError(
                    f"Weight delta v{version} not fully visible at {vdir}. The shared "
                    f"filesystem may need an explicit publish/refresh step "
                    f"(weight_sync_disk_pre_read_hook), or the trainer is "
                    f"publishing to a different disk_dir."
                )
            time.sleep(2.0)

    # ------------------------------------------------------------------
    # Loading patched tensors into the model
    # ------------------------------------------------------------------

    def _load_from_local(
        self,
        local_dir: str,
        update_info: DiskWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        locations = delta_codec.tensor_locations(local_dir)
        batch: list[tuple[str, torch.Tensor]] = []
        batch_nbytes = 0
        for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
            loc = locations.get(name)
            if loc is None:
                raise KeyError(
                    f"Tensor {name!r} not found in local checkpoint {local_dir} "
                    f"after applying v{update_info.version}"
                )
            raw = delta_codec.read_tensor_bytes(loc).copy()  # writable for from_numpy
            tensor = torch.from_numpy(raw).view(getattr(torch, dtype_name)).reshape(shape)
            batch.append((name, tensor))
            batch_nbytes += raw.nbytes
            if batch_nbytes >= _LOAD_BATCH_NBYTES:
                load_weights(batch)
                batch, batch_nbytes = [], 0
        if batch:
            load_weights(batch)
