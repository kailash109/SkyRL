"""Weight synchronization abstractions for distributed RL training.

Strategy classes are exported lazily (PEP 562): the broadcast and CUDA-IPC
strategies pull in ray and the trainer config stack, which must not load
inside vLLM worker processes that only need the disk receive path
(``disk_transfer_engine`` imports ``delta_codec`` through this package).
"""

from typing import Type

from .base import LoraLoadRequest, WeightChunk, WeightUpdateRequest
from .transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferSender,
    WeightTransferStrategy,
)
from .weight_extractor import WeightExtractor

_LAZY_EXPORTS = {
    "BroadcastInitInfo": "broadcast_strategy",
    "BroadcastTransferStrategy": "broadcast_strategy",
    "BroadcastWeightTransferSender": "broadcast_strategy",
    "BroadcastWeightUpdateRequest": "broadcast_strategy",
    "CudaIpcInitInfo": "cuda_ipc_strategy",
    "CudaIpcTransferStrategy": "cuda_ipc_strategy",
    "CudaIpcWeightTransferSender": "cuda_ipc_strategy",
    "CudaIpcWeightUpdateRequest": "cuda_ipc_strategy",
    "DiskInitInfo": "disk_strategy",
    "DiskTransferStrategy": "disk_strategy",
    "DiskWeightTransferSender": "disk_strategy",
    "DiskWeightUpdateRequest": "disk_strategy",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def get_transfer_strategy_cls(weight_sync_backend: str, colocate_all: bool) -> Type[WeightTransferStrategy]:
    """Get the appropriate transfer strategy class based on config.

    - ``"disk"``: disk-based delta sync (non-colocated only).
    - ``"nccl"`` + colocated: CUDA IPC (same GPUs, zero-copy).
    - otherwise: NCCL broadcast.

    Args:
        weight_sync_backend: The weight sync backend ("nccl" or "disk").
        colocate_all: Whether training and inference are colocated on same nodes.

    Returns:
        The strategy class.
    """
    strategy = get_transfer_strategy(weight_sync_backend, colocate_all)
    if strategy == "ipc":
        from .cuda_ipc_strategy import CudaIpcTransferStrategy

        return CudaIpcTransferStrategy
    if strategy == "disk":
        from .disk_strategy import DiskTransferStrategy

        return DiskTransferStrategy
    from .broadcast_strategy import BroadcastTransferStrategy

    return BroadcastTransferStrategy


def get_transfer_strategy(weight_sync_backend: str, colocate_all: bool) -> str:
    """Get the appropriate transfer strategy string based on config."""
    if weight_sync_backend == "disk":
        if colocate_all:
            raise ValueError(
                "weight_sync_backend='disk' is for non-colocated setups; "
                "use the default 'nccl' (CUDA IPC) when colocate_all=true"
            )
        return "disk"
    if weight_sync_backend == "nccl" and colocate_all:
        return "ipc"
    return "nccl"


__all__ = [
    "WeightChunk",
    "WeightExtractor",
    "WeightUpdateRequest",
    "LoraLoadRequest",
    "BroadcastWeightUpdateRequest",
    "CudaIpcWeightUpdateRequest",
    "WeightTransferStrategy",
    "WeightTransferSender",
    "WeightSyncInitInfo",
    "BroadcastInitInfo",
    "CudaIpcInitInfo",
    "BroadcastTransferStrategy",
    "BroadcastWeightTransferSender",
    "CudaIpcTransferStrategy",
    "CudaIpcWeightTransferSender",
    "DiskInitInfo",
    "DiskTransferStrategy",
    "DiskWeightTransferSender",
    "DiskWeightUpdateRequest",
    "get_transfer_strategy_cls",
]
