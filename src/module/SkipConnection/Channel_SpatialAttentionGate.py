import numpy as np
import torch.nn as nn
from AttentionGate import Attention_block
from CBAM import CBAM
class Channel_SpatialAttentionGate(nn.module):
    def __init__(self, in_channel, skip_connection, out_channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor = 2)
        self.cbam = CBAM()
        self.attgate = Attention_block(in_channel, skip_connection, skip_connection//2)
    def forward(x, skip):
        x = self.up(x)
        skip = self.cbam(skip)
        skip = self.attgate(x, skip)
        x = torch.cat([x,skip], dim =1)
        return x
