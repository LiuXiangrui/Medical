import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.head = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.res_blocks = nn.Sequential(*[ResBlock(channels=64) for _ in range(16)])

        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.head(x)
        f = self.res_blocks(f)
        return x + self.tail(f)


if __name__ == "__main__":
    a = Network()
    b = a(torch.randn(1, 1, 128, 128))
    print(b.shape)
