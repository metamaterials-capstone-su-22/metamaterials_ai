from einops.layers.torch import Rearrange
from torch import nn

from models.model_config import ModelConfig

from .mixer_module import MLPMixer


class MLPMixerArc:
    @staticmethod
    def create_model(model_config: ModelConfig):
        return (
            MLPMixerArc.create_forward_model(model_config)
            if model_config.direction == "forward"
            else MLPMixerArc.create_backward_model(model_config)
        )

    @staticmethod
    def create_forward_model(model_config: ModelConfig):
        return nn.Sequential(
            Rearrange("b c -> b c 1 1"),
            MLPMixer(
                in_channels=14,
                image_size=1,
                patch_size=1,
                num_classes=model_config.num_classes,
                dim=512,
                depth=8,
                token_dim=256,
                channel_dim=2048,
                dropout=0.5,
            ),
            nn.Sigmoid(),
        )

    @staticmethod
    def create_backward_model(model_config: ModelConfig):
        return nn.Sequential(
            Rearrange("b c -> b c 1 1"),
            MLPMixer(
                in_channels=model_config.in_channels,
                image_size=1,
                patch_size=1,
                num_classes=1_000,
                dim=512,
                depth=8,
                token_dim=256,
                channel_dim=2048,
                dropout=0.5,
            ),
            nn.Flatten(),
        )
