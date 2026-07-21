"""Tinker sampling client for an external Stitch SGLang rollout pool."""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

import httpx

from skyrl.backends.renderer import render_model_input
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.extra.external_inference import ExternalInferenceClientBase

if TYPE_CHECKING:
    from skyrl.tinker.api import SampleRequest


class ExternalStitchInferenceClient(ExternalInferenceClientBase):
    """Forwards Tinker samples to Stitch's version-gated SGLang endpoint."""

    def __init__(self, engine_config: EngineConfig, db_engine):
        super().__init__(db_engine)
        headers = {}
        if engine_config.external_inference_api_key:
            headers["Authorization"] = f"Bearer {engine_config.external_inference_api_key}"
        self._client = httpx.AsyncClient(
            base_url=engine_config.external_inference_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(engine_config.stitch_request_timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
        )
        self.max_retries = engine_config.stitch_max_retries
        self.retry_backoff_s = engine_config.stitch_retry_backoff_s

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post_generate(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post("/generate", json=payload, headers=headers)
                if response.status_code != 409 and response.status_code < 500:
                    response.raise_for_status()
                    return response.json()
            except httpx.RequestError:
                if attempt + 1 == self.max_retries:
                    raise
            if attempt + 1 < self.max_retries:
                await asyncio.sleep(self.retry_backoff_s)
        response.raise_for_status()
        raise RuntimeError("Stitch rollout request exhausted its retry budget")

    async def _forward_to_engine(
        self,
        request: "SampleRequest",
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
        weight_version: int | None = None,
    ) -> types.SampleOutput:
        del checkpoint_id
        if base_model is None and weight_version is None:
            raise ValueError("Sampler checkpoint has no Stitch weight version")

        prompt_tokens = render_model_input([request.prompt.to_types()])[0].prompt_ids
        version = 0 if base_model else int(weight_version)
        session_id = types.make_routing_session_id(request.sampling_session_id, request.seq_id)
        headers = {"X-Session-Affinity": session_id} if session_id else {}
        seed = request.sampling_params.seed
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        async def generate(sample_index: int) -> types.GeneratedSequence:
            params: dict[str, Any] = {
                "max_new_tokens": request.sampling_params.max_tokens,
                "temperature": request.sampling_params.temperature,
                "top_p": request.sampling_params.top_p,
                "top_k": request.sampling_params.top_k,
                "seed": seed + sample_index,
            }
            stop = request.sampling_params.stop
            if stop and isinstance(stop[0], int):
                params["stop_token_ids"] = stop
            elif stop:
                params["stop"] = stop

            payload: dict[str, Any] = {
                "input_ids": prompt_tokens,
                "sampling_params": params,
                "return_logprob": True,
                "weight_version": {"exact_version": version},
            }
            if base_model is None:
                payload["lora_path"] = model_id

            result = await self._post_generate(payload, headers)
            meta = result["meta_info"]
            start_version = int(meta["weight_version_start"])
            end_version = int(meta["weight_version_end"])
            if start_version != version or end_version != start_version:
                raise RuntimeError(f"Stitch served weight versions {start_version}->{end_version}, expected {version}")

            token_logprobs = meta.get("output_token_logprobs") or []
            finish_reason = meta.get("finish_reason") or {}
            finish_type = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
            return types.GeneratedSequence(
                tokens=[item[1] for item in token_logprobs],
                logprobs=[item[0] for item in token_logprobs],
                stop_reason="length" if finish_type == "length" else "stop",
            )

        sequences = await asyncio.gather(*(generate(i) for i in range(request.num_samples)))
        return types.SampleOutput(sequences=sequences, prompt_logprobs=[])
