from torch import nn


class SimpleBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.block = nn.Sequential(nn.Linear(dim, dim), nn.SELU())
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.block(x)
        x = self.dropout(x)
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
