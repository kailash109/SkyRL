# Weight Sync

Training-to-inference weight transfer. Runs after every training step (or on the configured interval) to push updated policy weights from training workers (FSDP/Megatron) into the vLLM inference engines.

## Architecture

Two-sided protocol with sender (training) / receiver (inference):

```
skyrl/backends/skyrl_train/weight_sync/
├── base.py                 # WeightUpdateRequest, LoraLoadRequest, WeightChunk
├── transfer_strategy.py    # WeightSyncInitInfo / Sender / Strategy ABCs (sender-side only; receive is vLLM-native)
├── broadcast_strategy.py   # NCCL broadcast (non-colocated)
├── cuda_ipc_strategy.py    # CUDA IPC (colocated)
├── disk_strategy.py        # Disk delta sync sender (non-colocated, shared FS)
├── delta_codec.py          # Byte-level xor/overwrite + zstd codec, safetensors patching (CPU-only)
├── weight_extractor.py     # Sharded-param -> dense tensor extraction
└── weight_extractor_utils.py
```

`weight_sync/__init__.py` exports the strategy classes **lazily** (PEP 562): the broadcast/IPC strategies import ray and the trainer config stack, which must not load inside vLLM worker processes that import `delta_codec` through the package. Keep new heavy imports out of the eager section.

vLLM worker-extension class (loaded via `--worker-extension-cls`):

- `skyrl/backends/skyrl_train/inference_servers/new_inference_worker_wrap.py` — `NewInferenceWorkerWrap`. Three-phase chunked lifecycle.

The weight sync implementation relies on the native vLLM weight sync APIs - `WeightTransferEngine` abstractions as well as native RPC endpoints for weight updates.

## Transfer Strategies

- **Broadcast** (`BroadcastTransferStrategy`): NCCL collective. Used for **non-colocated** setups. Training and inference are on different GPUs; weights cross the wire over a dedicated process group.
- **CUDA IPC** (`CudaIpcTransferStrategy`): Per-chunk packed buffer + one IPC handle per rank. Used for **colocated** setups (`colocate_all=true`). Both sides live on the same GPU; the receiver maps the sender's CUDA allocation directly.
- **Disk delta** (`DiskTransferStrategy`, `weight_sync_backend="disk"`): ships only the zstd-compressed bytes that changed since the last sync over a shared filesystem (`weight_sync_disk_dir`). Non-colocated only; for cross-cluster/DC setups or when a shared FS is the natural transport. Sender (rank 0) keeps a CPU base snapshot seeded from the HF checkpoint and publishes `v{N}/` delta dirs; the receiver is SkyRL's own `DiskWeightTransferEngine` (a custom vLLM `WeightTransferEngine`, registered under backend `"disk"` at `new_inference_worker_wrap.py` import), which patches a host-local checkpoint copy in place (file-lock, once per host) and reloads the patched tensors through `model.load_weights` — so TP/PP resharding works like the NCCL path. Design follows slime's delta weight sync. Config knobs: `weight_sync_disk_dir` (required, fresh dir per run), `weight_sync_local_ckpt_dir`, `weight_sync_delta_encoding` (`xor`/`overwrite`), `weight_sync_delta_checksum`, `weight_sync_disk_pre_read_hook` (`"module:function"` refresh callback for object-store-backed mounts whose writes aren't immediately visible across hosts — e.g. `modal.Volume.reload()`; POSIX shared FS doesn't need it).

Strategy choice is decided by the sender (`get_transfer_strategy_cls`). The init info is expanded per server via `for_servers()` / `to_api_payload()` and pushed to the servers through the HTTP control plane (`init_weight_update_communicator` → vLLM's native `/init_weight_transfer_engine`); the receive side is vLLM's native weight-transfer engine (or SkyRL's disk engine), driven by `NewInferenceWorkerWrap`.

## Lifecycle (`NewInferenceWorkerWrap`)
1. `start_weight_update(is_checkpoint_format=True)` — initializes layerwise reload (moves layers to meta device, wraps loaders).
2. `update_weights_chunk(update_info)` — called repeatedly. Unpacks the SkyRL packed CUDA-IPC payload, slices the contiguous buffer per param, calls `model.load_weights(weights=...)` under `set_current_vllm_config`.
3. `finish_weight_update()` — runs `finalize_layerwise_reload` (quantization repacking, attention weight postprocessing).

## Convention: vLLM imports

`vllm` is a Linux-only optional dep. Import it **lazily inside methods**, not at module top. Match the existing pattern in `new_inference_worker_wrap.py`.

## Tests

```bash
# CPU — chunk packing, transfer strategy unit tests, delta codec + disk sender
uv run --extra dev pytest tests/backends/skyrl_train/weight_sync/ -v

# GPU — end-to-end weight sync (NCCL + CUDA IPC paths, TP=1 and TP=2)
uv run --isolated --extra dev --extra fsdp \
  pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync.py -v

# Disk delta — two-node E2E on Modal (trainer + TP=2 vLLM, shared Volume)
modal run tests/backends/skyrl_train/gpu/modal/test_disk_weight_sync_modal.py
```

The CPU tests do **not** import `NewInferenceWorkerWrap`. Any change to the worker-extension class must be exercised by the GPU test above.

## When to touch what

| Change | Run |
|--------|-----|
| `WeightChunk` packing / size accounting | `tests/backends/skyrl_train/weight_sync/test_weight_chunk.py` |
| Broadcast or CUDA IPC sender | `test_transfer_strategies.py` (CPU) **and** GPU `test_weight_sync.py` |
| `NewInferenceWorkerWrap` | GPU `test_weight_sync.py` (CPU tests will not catch regressions) |

## vLLM version coupling

`vllm` is pinned in `pyproject.toml`. Weight-sync code paths are tightly coupled to vLLM internals (`model_runner.load_weights`, `initialize_layerwise_reload`, `SKIP_TENSORS`). When bumping the pin, re-verify the GPU weight-sync tests.

## Gotchas

- NemotronH / Mamba: vLLM's layerwise reload corrupts `conv1d.weight` via shared-storage view buffers. Workaround at the top of `new_inference_worker_wrap.py` adds `"conv_weights"` to `SKIP_TENSORS` at import time. Remove pending vLLM PR #42481 (vLLM 0.21.0).
- After `update_weights_chunk` runs, call `torch.accelerator.synchronize()` before returning so the sender doesn't drop its packed buffer mid-copy on the next barrier.
