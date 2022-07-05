from einops.layers.torch import Rearrange
from torch import nn

from models.model_config import ModelConfig

from .net2d_module import BasicConvBlock, ResNet


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
            Rearrange("b c -> b c 1 1"),
            ResNet(
                block_type=BasicConvBlock,
                num_blocks=[9, 9, 9],
                in_channels=14,
                out_channels=512,
            ),
            nn.Sigmoid(),
        )

        # in_channels=14,
        #         image_size=1,
        #         patch_size=1,
        #         num_classes=model_config.num_classes,
        #         dim=512,
        #         depth=8,
        #         token_dim=256,
        #         channel_dim=2048,
        #         dropout=0.5,

    @staticmethod
    def create_inverse_model(model_config: ModelConfig):
        return nn.Sequential(
            Rearrange("b c -> b c 1 2"),
            ResNet(
                block_type=BasicConvBlock,
                num_blocks=[9, 9, 9],
                in_channels=model_config.in_channels,
            ),
            nn.Flatten(),
        )
