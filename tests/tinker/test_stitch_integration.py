import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel
from sqlmodel import Session, SQLModel
from stitch.protocol import WeightVersionPolicy
from stitch.sync import PolicyViolation

from examples.tinker.stitch.provider import (
    MultiRunLoraSyncManager,
)
from skyrl.tinker import types
from skyrl.tinker.api import ModelInput, SampleRequest, SamplingParams
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    CheckpointDB,
    CheckpointStatus,
    ModelDB,
    SamplerStateDB,
    SamplerVersionDB,
    SessionDB,
)
from skyrl.tinker.extra.stitch_inference import ExternalStitchInferenceClient
from skyrl.tinker.stitch import StitchPublisher


def _config(tmp_path: Path, **updates) -> EngineConfig:
    values = {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "backend": "fsdp",
        "database_url": f"sqlite:///{tmp_path / 'tinker.db'}",
        "external_inference_provider": "stitch",
        "external_inference_url": "https://rollout.example",
        "stitch_bulletin_root": tmp_path / "bulletin",
        "stitch_max_retries": 2,
        "stitch_retry_backoff_s": 0,
    }
    values.update(updates)
    return EngineConfig(**values)


def _sample_request() -> SampleRequest:
    return SampleRequest(
        prompt=ModelInput(chunks=[{"type": "encoded_text", "tokens": [1, 2]}]),
        sampling_params=SamplingParams(max_tokens=4, seed=7),
        model_path="tinker://model-a/sampler_weights/checkpoint-a",
        num_samples=1,
    )


@pytest.mark.asyncio
async def test_stitch_client_retries_and_parses_version(tmp_path):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(409, json={"error": {"type": "WeightVersionNotReady"}})
        return httpx.Response(
            200,
            json={
                "text": "ok",
                "meta_info": {
                    "output_token_logprobs": [[-0.2, 11, None], [-0.3, 12, None]],
                    "weight_version_start": 3,
                    "weight_version_end": 3,
                    "finish_reason": {"type": "stop"},
                },
            },
        )

    client = ExternalStitchInferenceClient(_config(tmp_path), db_engine=None)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="https://rollout.example",
        transport=httpx.MockTransport(handler),
    )
    try:
        output = await client._forward_to_engine(
            _sample_request(),
            "model-a",
            "checkpoint-a",
            weight_version=3,
        )
    finally:
        await client.aclose()

    assert len(requests) == 2
    assert requests[0]["lora_path"] == "model-a"
    assert requests[0]["weight_version"] == {"exact_version": 3}
    assert output.sequences[0].tokens == [11, 12]
    assert output.sequences[0].logprobs == [-0.2, -0.3]


def test_stitch_publisher_writes_run_partitioned_manifest(tmp_path):
    publisher = StitchPublisher(tmp_path)
    version_dir = publisher.version_dir("model-a", 1)
    (version_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (version_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
                "r": 32,
                "lora_alpha": 32,
                "target_modules": ["q_proj"],
            }
        )
    )

    publisher.publish("model-a", 1, version_dir)

    assert publisher.board.read_latest("model-a") == 1
    manifest = publisher.board.read_manifest("model-a", 1)
    assert manifest.run_id == "model-a"
    assert manifest.version == 1
    assert manifest.backend == "lora"


class _RolloutEngine:
    backend = "lora"

    def __init__(self):
        self.loaded = {}

    async def flush_cache(self):
        pass

    async def pause_generation(self):
        pass

    async def continue_generation(self):
        pass

    async def unload(self, run_id):
        self.loaded.pop(run_id, None)

    async def apply_manifest(self, manifest, version_path):
        del version_path
        self.loaded[manifest.run_id] = manifest.version


@pytest.mark.asyncio
async def test_multi_run_manager_loads_requested_chain(tmp_path):
    publisher = StitchPublisher(tmp_path)
    version_dir = publisher.version_dir("model-a", 1)
    (version_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (version_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "base",
                "r": 8,
                "lora_alpha": 8,
                "target_modules": ["q_proj"],
            }
        )
    )
    publisher.publish("model-a", 1, version_dir)

    engine = _RolloutEngine()
    manager = MultiRunLoraSyncManager(publisher.board, engine, max_hot_chains=2)
    with pytest.raises(PolicyViolation):
        async with manager.request_context(WeightVersionPolicy(exact_version=1), run_id="model-a"):
            pass
    assert manager._sync_task is not None
    await manager._sync_task
    async with manager.request_context(WeightVersionPolicy(exact_version=1), run_id="model-a") as version:
        assert version == 1

    assert engine.loaded == {"model-a": 1}


class _BackendConfig(BaseModel):
    pass


class _Backend:
    def __init__(self, base_model: str, config: _BackendConfig):
        del base_model, config

    def has_model(self, model_id: str) -> bool:
        return model_id in {"model-a", "model-b"}

    def export_lora_adapter(self, output_dir: Path, model_id: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_model.safetensors").write_bytes(model_id.encode())
        (output_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "base",
                    "r": 8,
                    "lora_alpha": 8,
                    "target_modules": ["q_proj"],
                }
            )
        )

    def set_inference_state_publisher(self, publisher) -> None:
        del publisher


def test_tinker_engine_versions_are_per_model(tmp_path, monkeypatch):
    import skyrl.tinker.engine as engine_module

    monkeypatch.setattr(
        engine_module,
        "get_backend_classes",
        lambda backend_name, use_ray=False: (_Backend, _BackendConfig),
    )
    engine = engine_module.TinkerEngine(_config(tmp_path))
    SQLModel.metadata.create_all(engine.db_engine)

    with Session(engine.db_engine) as session:
        session.add(SessionDB(session_id="session", sdk_version="test"))
        for model_id in ("model-a", "model-b"):
            session.add(
                ModelDB(
                    model_id=model_id,
                    base_model="base",
                    lora_config={},
                    status="created",
                    request_id=1,
                    session_id="session",
                )
            )
            for checkpoint_id in ("checkpoint-1", "checkpoint-2"):
                session.add(
                    CheckpointDB(
                        model_id=model_id,
                        checkpoint_id=checkpoint_id,
                        checkpoint_type=types.CheckpointType.SAMPLER,
                        status=CheckpointStatus.PENDING,
                    )
                )
        session.commit()

    def request(checkpoint_id: str) -> types.SaveWeightsForSamplerInput:
        return types.SaveWeightsForSamplerInput(
            path=checkpoint_id,
            sampling_session_seq_id=1,
            seq_id=1,
        )

    engine.process_save_weights_for_sampler("model-a", request("checkpoint-1"))
    engine.process_save_weights_for_sampler("model-b", request("checkpoint-1"))
    engine.process_save_weights_for_sampler("model-a", request("checkpoint-2"))
    engine.process_save_weights_for_sampler("model-a", request("checkpoint-2"))

    with Session(engine.db_engine) as session:
        assert session.get(SamplerStateDB, "model-a").latest_published_version == 2
        assert session.get(SamplerStateDB, "model-b").latest_published_version == 1
        assert session.get(SamplerVersionDB, ("model-a", "checkpoint-1")).weight_version == 1
        assert session.get(SamplerVersionDB, ("model-a", "checkpoint-2")).weight_version == 2
        assert session.get(SamplerVersionDB, ("model-b", "checkpoint-1")).weight_version == 1
