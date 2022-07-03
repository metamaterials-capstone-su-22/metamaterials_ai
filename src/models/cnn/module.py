from torch import nn


class SimpleCnnBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int | tuple = 1) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size),
            nn.BatchNorm1d(dim),
            nn.SELU(),
        )

    def forward(self, x):
        x = x + self.block(x)
        return x


class BlocksBuilder(nn.Module):
    def __init__(self, num_blocks: int, dim: int) -> None:
        super().__init__()
        embedding_dim = 512
        n_filters = 10
        size = 1
        filter_sizes = 1
        self.convs = nn.ModuleList([SimpleCnnBlock(dim) for _ in range(num_blocks)])

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return x
