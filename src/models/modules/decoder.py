import torch
import torch.nn as nn

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
class AxialMixer(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.mixer_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.mixer_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)
        # self.dw = nn.Conv2d(dim, dim, kernel_size = 3, padding = 2, dilation = 2, groups = dim)

    def forward(self, x):
        # x = self.mixer_h(x) + self.mixer_w(x) + self.dw(x)
        x = x + self.mixer_h(x) + self.mixer_w(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
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
class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor = 2)
        self.att = Attention_block(F_g = in_c, F_l = skip_c, F_int= skip_c//2)
        self.residual = ResidualBlock(in_c + skip_c, out_c)
        # self.conv2 = nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding = 'same')
        # self.bn2 = nn.BatchNorm2d(out_c)
        # self.resmamba = ResMambaBlock(out_c)
        # self.act = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)

        x = torch.cat([x, skip], dim=1)
        # x = self.act(self.bn2(self.conv2(x)))
        x = self.residual(x)
        # x = self.resmamba(x)

        return x