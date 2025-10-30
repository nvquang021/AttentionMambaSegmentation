import torch
import torch.nn as nn
from models.modules.ScaleAttention import CBAM

class Attention_block(nn.Module):
    """
    Attention Gate (AG) module from Attention U-Net.
    Used to refine skip connections by suppressing irrelevant spatial regions.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g (int): number of channels in gating signal (from deeper layer)
            F_l (int): number of channels in skip connection (from encoder)
            F_int (int): number of intermediate channels (usually F_l // 2)
        """
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
class CSAG(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(CSAG, self).__init__()
        self.cbam = CBAM()