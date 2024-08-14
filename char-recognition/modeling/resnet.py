import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return F.relu(x + y)


class ResNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        img_size: tuple,
        num_classes: int,
        pretrained: bool,
    ):
        super().__init__()

        self.resnet18 = torch.hub.load(
            "pytorch/vision:v0.10.0", model_name, pretrained=pretrained
        )
        self.fc = nn.Linear(1000, num_classes)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.residual_blocks = nn.Sequential([

        # ])

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * 1 * 1, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet18(x)
        x = self.fc(x)
        return x
