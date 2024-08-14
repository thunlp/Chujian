from torch import nn, Tensor
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.dropout(y, 0.2)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return F.relu(x + y)


class ResNet(nn.Module):
    def __init__(
        self,
        img_size: tuple,
        num_classes: int,
        pretrained_name: str = 'DEFAULT',
        pretrained: bool = True,
        dim: int = 32
    ):
        super().__init__()

        # self.resnet18 = torch.hub.load(
        #     "pytorch/vision:v0.10.0", pretrained_name, pretrained=pretrained
        # )
        # self.fc = nn.Linear(1000, num_classes)

        self.dim = dim
        self.in_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_conv = nn.Conv2d(3, dim, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

        print("image size:", img_size)

        res_blocks = [
            ResidualBlock(dim, dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(dim, dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(dim, dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(dim, dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ResidualBlock(dim, dim),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        repr_dim = dim * img_size[0] * img_size[1] // (4 ** 5)
        print('repr_dim:', repr_dim)
        cls_dim = 256

        self.fc1 = nn.Linear(repr_dim, cls_dim)
        self.fc2 = nn.Linear(cls_dim, num_classes)
        self.residual_blocks = nn.Sequential(*res_blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_pool(x)
        x = self.in_conv(x)
        # x = self.resnet18(x)
        x = self.residual_blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
