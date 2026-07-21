from skyrl.tinker.extra.external_inference import ExternalInferenceClient
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)
from skyrl.tinker.extra.stitch_inference import ExternalStitchInferenceClient

__all__ = [
    "ExternalInferenceClient",
    "ExternalStitchInferenceClient",
    "SkyRLTrainInferenceForwardingClient",
]
