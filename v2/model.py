"""
This is unofficial implementation of XuNet: Structural Design of Convolutional
Neural Networks for Steganalysis . """
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class ImageProcessing(nn.Module):
    """Computes convolution with KV filter over the input tensor."""
    def __init__(self) -> None:
        super().__init__()
        kv = torch.tensor([
            [-1.0, 2.0, -2.0, 2.0, -1.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-2.0, 8.0, -12.0, 8.0, -2.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-1.0, 2.0, -2.0, 2.0, -1.0],
        ]).view(1, 1, 5, 5) / 12.0
        self.register_buffer('kv_filter', kv)

    def forward(self, inp: Tensor) -> Tensor:
        return F.conv2d(inp, self.kv_filter, stride=1, padding=2)


class ConvBlock(nn.Module):
    """This class returns building block for XuNet class."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        abs: str = False,
    ) -> None:
        super().__init__()

        if kernel_size == 5:
            self.padding = 2
        else:
            self.padding = 0

        if activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        self.abs = abs
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=self.padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns conv->batch_norm."""
        if self.abs:
            return self.pool(
                self.activation(self.batch_norm(torch.abs(self.conv(inp))))
            )
        return self.pool(self.activation(self.batch_norm(self.conv(inp))))


class XuNet(nn.Module):
    """XuNet model for steganalysis."""
    def __init__(self) -> None:
        super().__init__()
        self.preprocessing = ImageProcessing()
        self.layer1 = ConvBlock(1, 8, kernel_size=5, activation="tanh", abs=True)
        self.layer2 = ConvBlock(8, 16, kernel_size=5, activation="tanh")
        self.layer3 = ConvBlock(16, 32, kernel_size=1)
        self.layer4 = ConvBlock(32, 64, kernel_size=1)
        self.layer5 = ConvBlock(64, 128, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(in_features=128, out_features=2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        # Allow gradients through preprocessing
        out = self.preprocessing(image)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected(out)
        return out


if __name__ == "__main__":
    net = XuNet()
    print(net)
    inp_image = torch.randn((1, 1, 512, 512))
    print(net(inp_image))

if __name__ == "__main__":
    # Test the ZhuNet model with a random input
    model = XuNet()
    model.eval()
    # get time to run one forward pass
    input_tensor = torch.randn(1, 1, 256, 256)  # Example input tensor
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be [1, num_classes]
    print("Model architecture:\n", model)
    import time
    start_time = time.time()
    output = model(input_tensor)
    end_time = time.time()
    print("Time for one forward pass: {:.6f} seconds".format(end_time - start_time))
    # Print model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    print("Total model parameters:", total_params)
