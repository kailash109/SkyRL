# RFC: Stitch-disaggregated rollouts for the SkyRL Tinker server

**Status**: Draft
**Author**: kailash
**Date**: 2026-07-17

## Summary

Add an opt-in mode to the SkyRL Tinker server where `sample()` traffic is served by an
**elastic, remote rollout fleet** behind a public proxy instead of trainer-managed vLLM
engines, and weight updates are delivered via a **bulletin board** (versioned artifacts on
a shared volume + a monotonic `latest` pointer) instead of NCCL broadcast / CUDA-IPC RPCs.

The protocol is [stitch](https://github.com/-/stitch) (local: `~/stitch`): trainer publishes
immutable versioned weight artifacts and advances a pointer; each rollout engine sits behind
a sidecar that catches the engine up to `latest`, stamps every response with the weight
version that served it, and admission-gates requests that pin a version (`409
WeightVersionNotReady` = retryable). Rollout compute becomes fully decoupled from the
trainer: containers autoscale, come and go, and converge on the pointer without the trainer
tracking them.

## Motivation

Today the Tinker backend owns its inference fleet end to end:

- `sample()` fans out to a locally-created router + `VLLMServerActor` server groups
  (`skyrl/backends/skyrl_train_backend.py::_sample_with_remote_client`), created lazily on
  the first sampling call and colocated (or not) with training GPUs.
- Weight sync is synchronous and connection-oriented: `save_weights_for_sampler` →
  `broadcast_to_inference_engines` → NCCL / CUDA-IPC into vLLM's native weight-transfer
  engine, with pause/resume, sleep/wake, and prefix-cache choreography around it
  (`skyrl/backends/skyrl_train/workers/worker_dispatch.py::save_weights_for_sampler`).

This couples rollout capacity to the trainer's Ray cluster and its GPU topology. For
agentic / long-tail rollouts we want the opposite: rollout capacity that scales
independently (e.g. Modal Flash autoscaling), survives trainer restarts, and can serve
multiple consumers. It also removes the single most painful lifecycle coupling in the
current design: `delete_model` on the last tenant tears down the *entire* shared runtime
including `ray.shutdown()` (`skyrl_train_backend.py::delete_model`), because engines are
trainer-owned. With stitch, there is nothing to tear down.

## Background: the two systems

### Stitch protocol (canonical package: `~/stitch/src/stitch/`)

- **Bulletin board** (`bulletin.py::FilesystemBulletinBoard`): shared volume with
  `versions/weight_vNNNNNN/manifest.json` + `latest.json`. `advance(run_id, version)` is
  monotonic within a run (`protocol.py::decide_pointer_move`, raises `PointerRewind`); a new
  `run_id` forks at base and resets the version space (trainer-restart fencing).
- **`VersionManifest`** (`protocol.py:230`): `version`, `base_version`, `backend`,
  `artifacts[]` (path, checksum, size), `run_id`, `base_model`.
- **Sidecar** (`servers/sglang.py::create_app`): FastAPI proxy in front of one engine.
  Exposes `/health`, `/server_info`, `/get_weight_version`,
  `POST /rpc_sync_from_bulletin_board` (the wake RPC); 403-blocks direct engine
  weight-control routes. Per-request: parses a `weight_version` policy from the body
  (`exact_version` | `min_required_version`), admission-gates it
  (`sync.py::RolloutAdmissionGate`), namespaces the KV cache key by version, and stamps
  `weight_version_start/end` into the response.
- **Sync state machine** (`sync.py::WeightSyncManager`): converges the engine to the
  pointer via wake RPC, periodic poll, and startup sync; verifies version-chain contiguity;
  two commit modes (`quiesce` = drain + flush, `in_place` = pause + apply, KV isolation via
  version-namespaced cache keys).
- **Engine adapter** (`protocol.py::EngineAdapter` Protocol): `apply_manifest`,
  `flush_cache`, `pause_generation`, `continue_generation`, optional `prepare`/`reset`.
  SGLang adapter exists (`engines/sglang.py`); **vLLM adapter is the piece this RFC adds**.
- **Provider** (`providers/modal.py`): volume commit/reload, Flash container discovery
  (`discover_flash_targets`), best-effort wake fan-out (`wake_targets`).

### SkyRL seams we build on

1. **External proxy mode already exists.**
   `generator.inference_engine.external_proxy_url` collapses data plane *and* control plane
   onto one URL with no local server groups, no router, no placement group
   (`skyrl/backends/skyrl_train/inference_servers/setup.py::build_new_inference_client`,
   external-proxy branch). Stitch mode is this branch plus a request contract.
2. **Weight-sync already has three shapes**, dispatched in
   `worker_dispatch.py::save_weights_for_sampler`: colocated NCCL/IPC broadcast,
   non-colocated broadcast with pause/resume, and the **in-place LoRA disk path** (export
   adapter safetensors to a shared dir + `load_lora_adapter` on the engine —
   `megatron_worker.py::_save_lora_adapters_and_sync`). The bulletin publisher is a fourth
   shape that generalizes the third: same artifact, but versioned, manifested, and
   pointer-advanced instead of pushed to known engines.
3. **A monotonic weight version already exists** on the client
   (`remote_inference_client.py::increment_weight_version`, used as the prefix-cache salt).
   Stitch turns this from a local counter into the *shared* version number.
4. **Multi-tenancy maps 1:1**: a Tinker `model_id` (one LoRA tenant, one slot in the
   megatron `AdapterStore`) is exactly a stitch `run_id` (one version chain). The
   front door's `run_resolver` keys on the adapter name in the request payload.

## Design

### Mode selection

New config block under `generator.inference_engine`:

```yaml
generator:
  inference_engine:
    stitch:
      enabled: true
      front_door_url: https://…      # becomes external_proxy_url
      bulletin_root: /bulletin       # shared volume mount (trainer side)
      publish_format: lora_adapter   # phase 1: lora_adapter only
      version_policy: latest         # or {max_lag: k}
      wake_on_publish: true
      commit_mode: in_place          # forwarded to sidecar deployment docs
```

Validation: requires `run_engines_locally=false` and `colocate_all=false`; forbids
setting `weight_sync_backend` (there is no transfer connection).

### Data plane

`sample()` path is unchanged down to the HTTP client. Changes live in a new
`StitchInferenceClient(RemoteInferenceClient)`:

- **Version pinning**: inject `{"weight_version": {"min_required_version": N}}` into the
  generate body, where `N` is the last version this backend published for the request's
  `model_id` (`exact_version` when the Tinker request carries a `checkpoint_id` — see
  "Checkpoints as versions" below).
- **409 handling**: treat stitch's `WeightVersionNotReady` as retryable with backoff in
  `_post` (it means "container still catching up", expected for seconds after a publish).
- **Version metadata**: parse `weight_version_start/end` from responses and thread them
  onto each sampled sequence.
- **Control-plane no-ops**: `pause/resume`, `sleep/wake_up`, `reset_prefix_cache`,
  `init_weight_update_communicator`, `update_weights_*`, `load/unload_lora_adapter`
  become no-ops (sidecar 403-blocks them; KV isolation across versions is the sidecar's
  job via version-namespaced cache keys, not the trainer's).

### Weight plane: the bulletin publisher

New module `skyrl/backends/skyrl_train/weight_sync/bulletin_publisher.py`, wrapping
stitch's `FilesystemBulletinBoard` + `VersionManifest` (optional dependency, e.g.
`--extra stitch`). Fourth branch in `WorkerDispatch.save_weights_for_sampler`:

```
1. ensure_active_adapter(model_id)                        # unchanged
2. export adapter artifact -> <bulletin_root>/versions/weight_vN/
     megatron: export_adapter_weights + _convert_moe_experts_lora_to_vllm
     (same artifact as the existing merge_lora=False disk-sync path)
3. write VersionManifest{kind: lora_adapter, base_model, run_id=model_id, version=N}
4. board.advance(run_id, N)          # rank 0 only; monotonic, PointerRewind-safe
5. commit volume                     # durability point (Modal: providers/modal.py)
6. wake (best-effort): POST /rpc_sync_from_bulletin_board to discovered containers
7. record N as the model's current version (drives sample-time pinning)
```

No pause/resume, no engine wake/sleep, no `_finish_weight_sync` offload choreography —
none of the engines are on trainer GPUs.

**Why LoRA-adapter artifacts (phase 1)**: megabytes per version instead of the ~57 GB a
merged-weight publish would cost for Qwen3-30B; rollout-side "apply" is an adapter load,
not an engine reload; and the exporter already exists and is battle-tested by the
in-place LoRA sync path. Full-finetune support means adopting stitch's disk-delta
encoding and is explicitly out of scope for phase 1.

### Rollout containers (new, deployable independently of the trainer)

- **vLLM `EngineAdapter` for stitch** (contributed to stitch as `engines/vllm.py`, or
  carried in SkyRL until upstreamed). For `kind: lora_adapter`:
  - `apply_manifest(m)`: `POST /v1/load_lora_adapter {name: "wv<N>", path: <artifact>}`;
    unload the previous version's adapter after commit.
  - `pause_generation` / `continue_generation` / `flush_cache`: vLLM-native endpoints.
  - `prepare()`: download base model once; `reset()`: unload all adapters (run switch).
- **Deployment**: mirror `~/stitch/cookbook/standalone_rollouts/modal_serve.py` —
  `@app.cls` + `@modal.experimental.http_server` + `@modal.concurrent`, boot =
  `vllm serve <base> --enable-lora` + stitch sidecar subprocess, health = sidecar
  `/health` after `startup_sync()` converges. Autoscaling via `min_containers` +
  `scaledown_window`; a scaled-up container catches up from the pointer with no trainer
  involvement.
- The per-request `model` field for sampling is the versioned adapter name; the
  sidecar's `run_resolver` keys the admission gate and KV namespace on it.

### Tinker backend lifecycle changes (`skyrl_train_backend.py`)

- `_ensure_inference_engines`: stitch mode constructs `StitchInferenceClient`, skips
  colocate placement-group logic, skips the initial `sleep()`, and **skips
  `init_weight_sync_state`** (no transfer engine). First init instead runs
  `board.claim(run_id)` (or defers publishing to the first `save_weights_for_sampler`).
- `save_sampler_checkpoint`: the NCCL sync call is replaced by a publish. With
  `persist=True`, the artifact is *already durable on the volume* — persisting becomes
  recording the version, not a second export.
- `delete_model`: in stitch mode the "last model" branch no longer tears down server
  groups / router / Ray-inference state (there is none). Session expiry stops publishing;
  containers idle down on their own. This structurally removes the
  expire-session-→-full-runtime-teardown behavior.
- `_sample_with_remote_client`: resolve the per-request version pin from the model's
  last published version (replaces the adapter-name-vs-base-name routing logic on this
  path; the served model name is the versioned adapter).

### Checkpoints as versions (phase 2)

`save_weights_for_sampler(name=…)` currently produces a checkpoint tarball; the engine
constrains batches to one checkpoint per model (`skyrl/tinker/engine.py::
find_batchable_sample`). In stitch mode a sampler checkpoint **is** a bulletin version, so:

- `checkpoint_id` ↔ `weight_vN` aliasing makes `create_sampling_client(model_path=…)`
  an `exact_version` pin,
- the one-checkpoint-per-model batching constraint can relax: stitch serves mixed
  versions concurrently with KV isolation, so requests pinned to different versions of
  the same model can batch through the same fleet.

### Tinker API surface

Thread `weight_version_start/end` onto each sampled sequence (same pass-through pattern
as the `routing_matrix` field): clients need it for staleness-aware off-policy
correction, and the train-vs-rollout logprob-gap metrics must be interpreted jointly
with the serving version once samples can lag.

## Code-change inventory

| # | Location | Change | Size |
|---|---|---|---|
| 1 | `skyrl/train/config/config.py` | `StitchConfig` block + validation | S |
| 2 | `inference_servers/stitch_client.py` (new) | `StitchInferenceClient`: pinning, 409 retry, metadata, control-plane no-ops | M |
| 3 | `inference_servers/setup.py` | stitch branch (external-proxy branch constructing the subclass) | S |
| 4 | `weight_sync/bulletin_publisher.py` (new) | board/manifest wrapper + export + advance + commit + wake | M |
| 5 | `workers/worker_dispatch.py` | fourth `save_weights_for_sampler` branch | S |
| 6 | `skyrl_train_backend.py` | mode gating: `_ensure_inference_engines`, `save_sampler_checkpoint`, `delete_model`, `_sample_with_remote_client` | M |
| 7 | `skyrl/tinker/types.py`, `api.py` | `weight_version_start/end` on sequences | S |
| 8 | stitch repo (or `examples/` until upstreamed) | vLLM `EngineAdapter` (`lora_adapter` kind) | M |
| 9 | `examples/tinker/stitch/` (new) | Modal rollout-fleet deployment (vLLM + sidecar) + front door | M |

## Trade-offs

- **Sync semantics become async.** `save_weights_and_get_sampling_client` returns at
  publish-durability, not after every container has loaded the version. The first
  samples after a publish may eat a 409-retry window (seconds, for adapter-only loads).
  This is the deliberate stitch trade: elasticity over lockstep.
- **Staleness is now a first-class variable.** With `version_policy: latest` behavior
  approximates today's on-policy guarantee at the cost of retry latency; with a lag
  allowance, throughput improves but per-datum `weight_version` must feed the training
  metrics (and any router-replay analysis) or staleness confounds the logprob-gap signal.
- **Multi-step chain replay**: a container that slept through k versions replays k
  adapter loads (cheap) — but the artifact chain on the volume must be retained for the
  window; needs a GC policy keyed to the oldest live container version.
- **Publish size gates FFT.** LoRA-only in phase 1; full-finetune requires disk-delta
  encoding (stitch supports it; SkyRL-side exporter does not exist yet).

## Phasing

1. **Phase 1 — single tenant, LoRA, min-version pinning**: items 1-6, 8, 9; one run_id;
   `version_policy: latest`.
2. **Phase 2 — versions as checkpoints + metadata**: item 7; `exact_version` pins from
   `checkpoint_id`; per-datum version logging in metrics; relax one-checkpoint-per-model
   batching.
3. **Phase 3 — multi-tenant + FFT**: run-per-`model_id` chains via front-door
   `run_resolver`; disk-delta exporter for full-finetune.

## Open questions

1. **Where does the vLLM `EngineAdapter` live** — upstream in stitch (preferred; it's
   protocol-side) or vendored in SkyRL until stable?
2. **Base-model rollout before the first publish**: serve base (`weight_v0` = empty
   adapter chain) or block sampling until the first publish? Base-serving matches
   Tinker's current lazy-engine semantics.
3. **Volume choice off-Modal**: `FilesystemBulletinBoard` needs a shared FS; for
   non-Modal deployments (S3/GCS), does stitch grow an object-store board, or do we
   require FUSE mounts?
4. **Logprobs parity**: sidecar-fronted engines must return per-token logprobs on the
   generate route (the Tinker datum contract depends on it) — verify the stitch proxy
   passes `logprobs` params through untouched.
5. **Router-replay interaction**: `enable_return_routed_experts` is engine config and
   composes, but routing matrices captured on version N and replayed on version N+k are
   exactly the mismatch router replay is meant to *remove* — replay datums should carry
   an `exact_version` requirement on the training side.
