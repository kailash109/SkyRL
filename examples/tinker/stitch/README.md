# Tinker training with external Stitch rollouts

This example runs SkyRL training and rollout serving in separate Modal
containers:

- a four-GPU SkyRL Tinker server trains a Qwen3-4B LoRA policy;
- an elastic H100 pool serves the frozen base model with SGLang;
- SkyRL publishes each adapter version through a shared bulletin-board Volume;
- Stitch sidecars pull adapters and enforce exact-version sampling.

From the repository root:

```bash
uv run --isolated --extra tinker --with modal \
  modal run --detach examples/tinker/stitch/modal_run.py --steps 2
```

The SkyRL checkout must have the corresponding Stitch checkout at `../stitch`.
Set `STITCH_LOCAL_DIR` when launching Modal if it lives elsewhere. The rollout
sidecar requires Stitch's multi-run `run_resolver` support.

Training reads W&B credentials from the `wandb-secret` Modal secret:

```bash
modal secret create wandb-secret \
  WANDB_API_KEY=... \
  WANDB_PROJECT=skyrl-tinker-stitch
```

`WANDB_ENTITY` and `WANDB_RUN_NAME` may also be stored in the same secret.

The example creates these persistent Modal Volumes:

- `skyrl-tinker-stitch-bulletin`
- `skyrl-hf-cache`
- `skyrl-tinker-stitch-data`
- `skyrl-tinker-stitch-checkpoints`

The rollout pool uses one H100 per replica and scales from zero to four
replicas. The trainer uses four H100s. Adjust the GPU declarations and
container limits in `modal_run.py` for a different cluster budget.
