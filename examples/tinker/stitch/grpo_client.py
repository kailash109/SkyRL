"""Small GSM8K GRPO loop for the external Stitch rollout example."""

from __future__ import annotations

import argparse
import logging
import math
import os

import tinker
from tinker import types

from examples.tinker.ppo.ppo_client import (
    build_policy_train_datum,
    compute_gsm8k_reward,
    load_split,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    return parser.parse_args()


def group_advantages(rewards: list[float]) -> list[float]:
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    scale = math.sqrt(variance + 1e-8)
    return [(reward - mean) / scale for reward in rewards]


def main() -> None:
    args = parse_args()
    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    policy = service.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    tokenizer = policy.get_tokenizer()
    records = load_split(args.data, tokenizer, max_prompt_length=512)

    try:
        for step in range(args.steps):
            start = step * args.prompts_per_step
            batch = records[start : start + args.prompts_per_step]
            sampler = policy.save_weights_and_get_sampling_client(name=f"step_{step}")
            futures = [
                sampler.sample(
                    prompt=types.ModelInput.from_ints(record.prompt_tokens),
                    num_samples=args.samples_per_prompt,
                    sampling_params=types.SamplingParams(
                        max_tokens=args.max_tokens,
                        temperature=1.0,
                        top_p=1.0,
                    ),
                )
                for record in batch
            ]

            datums = []
            rewards = []
            for record, future in zip(batch, futures, strict=True):
                sequences = future.result().sequences
                group_rewards = [
                    compute_gsm8k_reward(
                        tokenizer.decode(sequence.tokens, skip_special_tokens=True),
                        record.ground_truth,
                    )
                    for sequence in sequences
                ]
                rewards.extend(group_rewards)
                for sequence, advantage in zip(sequences, group_advantages(group_rewards), strict=True):
                    datums.append(
                        build_policy_train_datum(
                            record.prompt_tokens,
                            list(sequence.tokens),
                            list(sequence.logprobs),
                            [advantage] * len(sequence.tokens),
                        )
                    )

            policy.forward_backward(
                datums,
                "ppo",
                {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
            ).result()
            result = policy.optim_step(types.AdamParams(learning_rate=args.learning_rate)).result()
            logger.info(
                "step=%s reward=%.3f trajectories=%s metrics=%s",
                step + 1,
                sum(rewards) / len(rewards),
                len(datums),
                result.metrics,
            )
    finally:
        service.holder.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
