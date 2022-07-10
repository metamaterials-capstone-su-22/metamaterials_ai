from einops.layers.torch import Rearrange
from torch import nn, torch

from models.model_config import ModelConfig
from einops import rearrange

# from .module import BlocksBuilder


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
        num_blocks: int = 16
        dim: int = 512
        size = 1
        return nn.Sequential(
            Rearrange("b c -> b c 1"),
            nn.Conv1d(
                in_channels=model_config.in_channels, out_channels=dim, kernel_size=size
            ),
            nn.BatchNorm1d(dim),
            nn.SELU(),
            BlocksBuilder(num_blocks=num_blocks, dim=dim),
            nn.Conv1d(
                in_channels=dim, out_channels=model_config.num_classes, kernel_size=size
            ),
            nn.SELU(),
            Rearrange("b c h -> (b h) c"),
            nn.Sigmoid(),
        )

    @staticmethod
    def create_inverse_model(model_config: ModelConfig):
        num_blocks: int = 2
        conv_dim: int = 9
        fc_dim: int = 512
        kernel_size = 5  # TODO: Fix this => model_config.kernel_size
        pool_size = 2
        filter_sizes = 1
        fc_input = conv_dim * 96

        return nn.Sequential(
            Rearrange("b c -> b 1 c"),
            nn.Conv1d(in_channels=1, out_channels=conv_dim,
                      kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.BatchNorm1d(conv_dim),
            nn.SELU(),
            BlocksBuilder(num_blocks=num_blocks, dim=conv_dim,
                          kernel_size=kernel_size, pool_size=pool_size),
            nn.Flatten(),
            nn.Sequential(nn.Linear(fc_input, fc_dim), nn.SELU()),
            AnnBlocksBuilder(num_blocks=15, dim=fc_dim),
            nn.Sequential(nn.Linear(fc_dim, 14), nn.SELU()),
        )

    @staticmethod
    def gauss_kernel(n=5, sigma=1):
        r = range(-int(n/2), int(n/2)+1)
        kernel = torch.FloatTensor(
            [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r])
        return rearrange(kernel, 'w -> 1 1 w')


class SimpleCnnBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int | tuple = 1, pool_size: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=dim,
                      kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.BatchNorm1d(dim),
            nn.SELU(),
        )

    def forward(self, x):
        # x = x + self.block(x)
        x = self.block(x)
        return x


class SimpleAnnBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.block = nn.Sequential(nn.Linear(dim, dim), nn.SELU())
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.block(x)
        x = self.dropout(x)
        return x


class BlocksBuilder(nn.Module):
    def __init__(self, num_blocks: int, dim: int, kernel_size: int | tuple = 1, pool_size: int = 1) -> None:
        super().__init__()
        self.convs = nn.ModuleList([SimpleCnnBlock(dim, kernel_size=kernel_size,
                                                   pool_size=pool_size)
                                    for _ in range(num_blocks)])

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return x


class AnnBlocksBuilder(nn.Module):
    def __init__(self, num_blocks: int, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SimpleAnnBlock(dim)
                                    for _ in range(num_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
