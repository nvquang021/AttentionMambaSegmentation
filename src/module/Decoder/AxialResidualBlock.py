import torch
import torch.nn as nn

class AxialResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(AxialResidualBlock, self).__init__()
        self.conv1 = AxialMixer(in_channels, mixer_kernel=(3, 3))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv3 = AxialMixer(out_channels, mixer_kernel=(3,3))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out