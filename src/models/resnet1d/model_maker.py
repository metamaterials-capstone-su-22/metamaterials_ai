from einops.layers.torch import Rearrange
from torch import nn

from models.model_config import ModelConfig

from .net1d_module import ResNet1D


class ModelMaker:
    @staticmethod
    def create_model(model_config: ModelConfig):
        return (
            ModelMaker.create_direct_model(model_config)
            if model_config.direction == "direct"
            else ModelMaker.create_inverse_model(model_config)
        )

    @staticmethod
    def create_direct_model(model_config: ModelConfig):
        return nn.Sequential(
            Rearrange("b c -> b c 1"),
            ResNet1D(
                in_channels=14,
                base_filters=1,
                kernel_size=1,
                stride=2,
                groups=1,
                n_block=4,
                n_classes=model_config.num_classes,
                downsample_gap=2,
                increasefilter_gap=4,
                use_bn=True,
                use_do=True,
                verbose=False,
            ),
            nn.Sigmoid(),
        )

    @staticmethod
    def create_inverse_model(model_config: ModelConfig):
        return nn.Sequential(
            Rearrange("b c -> b c 1"),
            ResNet1D(
                in_channels=model_config.in_channels,
                base_filters=1,
                kernel_size=1,
                stride=2,
                groups=1,
                n_block=1,
                n_classes=1_000,
                downsample_gap=2,
                increasefilter_gap=4,
                use_bn=True,
                use_do=True,
                verbose=False,
            ),
            nn.Flatten(),
        )
