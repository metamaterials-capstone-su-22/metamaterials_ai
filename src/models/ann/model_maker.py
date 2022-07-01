from einops.layers.torch import Rearrange
from torch import nn

from models.model_config import ModelConfig


class ModelMaker:
    @staticmethod
    def create_model(model_config: ModelConfig):
        return (
            ModelMaker.create_forward_model(model_config)
            if model_config.direction == "forward"
            else ModelMaker.create_backward_model(model_config)
        )

    @staticmethod
    def create_forward_model(model_config: ModelConfig):
        num_blocks: int = 16
        dim: int = 512
        return nn.Sequential(
            nn.Linear(model_config.in_channels, dim),
            nn.SELU(),
            BlocksBuilder(num_blocks=num_blocks, dim=dim),
            nn.Linear(dim, model_config.num_classes),
            nn.Sigmoid(),
        )


class SimpleBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SELU()
        )

    def forward(self, x):
        x = x + self.block(x)
        return x


class BlocksBuilder(nn.Module):
    def __init__(self, num_blocks: int, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SimpleBlock(dim)
                                    for _ in range(num_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
