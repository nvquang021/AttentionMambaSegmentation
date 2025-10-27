from src.models.modules.encoder import NewEncoderBlock
from src.models.modules.decoder import DecoderBlock
import torch
import torch.nn as nn
from src.models.modules.skipconnection import CBAM
from src.models.modules.selective_kernels import SKConv_7, SKUnit
from src.models.modules.UnetDsv3 import UnetDsv3
from src.models.modules.skipconnection import scale_atten_convblock

class DA_MambaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pw_in = nn.Conv2d(3, 16, kernel_size=1)
        self.sk_in = SKConv_7(16, M=2, G=16, r=4, stride=1 ,L=32)
        """Encoder"""
        self.e1 = NewEncoderBlock(16, 32, 256, 192)
        self.e2 = NewEncoderBlock(32, 64, 128, 96)
        self.e3 = NewEncoderBlock(64, 128, 64, 48)
        self.e4 = NewEncoderBlock(128, 256, 32, 24)
        self.e5 = NewEncoderBlock(256, 512, 16, 12)

        """Skip connection"""
        self.s1 = CBAM(gate_channels = 32)
        self.s2 = CBAM(gate_channels = 64)
        self.s3 = CBAM(gate_channels = 128)
        self.s4 = CBAM(gate_channels = 256)
        self.s5 = CBAM(gate_channels = 512)

        """Bottle Neck"""
        # self.b5 = PCAPSA_Mamba(512)
        self.b5 = SKUnit(512, 512, 512, M=2, G=16, r=2, stride=1, L=32)
        """Decoder"""
        self.d5 = DecoderBlock(512, 512, 256)
        self.d4 = DecoderBlock(256, 256, 128)
        self.d3 = DecoderBlock(128, 128, 64)
        self.d2 = DecoderBlock(64, 64, 32)
        self.d1 = DecoderBlock(32, 32, 16)

        # self.out = OutBlock(3, 1)
         # deep supervision
        self.dsv5 = UnetDsv3(256, 4, scale_factor=(192,256))
        self.dsv4 = UnetDsv3(128,4, scale_factor=(192,256))
        self.dsv3 = UnetDsv3(64, 4, scale_factor=(192,256))
        self.dsv2 = UnetDsv3(32, 4, scale_factor=(192,256))
        self.dsv1 = UnetDsv3(16, 4, scale_factor=(192,256))
        self.scale_attention = scale_atten_convblock(16,4)
        self.conv_final = nn.Conv2d(16, 1, kernel_size=1)
        self.conv_out = nn.Conv2d(4, 1, kernel_size=1)
    def forward(self, x):
        """Encoder"""
        x = self.pw_in(x)
        x = self.sk_in(x)
        # x = self.qseme(x)
        x, skip1 = self.e1(x)

        x, skip2 = self.e2(x)

        x, skip3 = self.e3(x)

        x, skip4 = self.e4(x)

        x, skip5 = self.e5(x)

        """Skip connection"""
        skip1 = self.s1(skip1)

        skip2 = self.s2(skip2)

        skip3 = self.s3(skip3)

        skip4 = self.s4(skip4)

        skip5 = self.s5(skip5)
        """BottleNeck"""
        x = self.b5(x)         # (512, 8, 8)

        """Decoder"""
        x5 = self.d5(x, skip5)
        x4 = self.d4(x5, skip4)
        x3 = self.d3(x4, skip3)
        x2 = self.d2(x3, skip2)
        x1 = self.d1(x2, skip1)
        decoder_out = self.conv_final(x1)
        "Deep Supervision"
        dsv5 = self.dsv5(x5)
        dsv4 = self.dsv4(x4)
        dsv3 = self.dsv3(x3)
        dsv2 = self.dsv2(x2)
        dsv1 = self.dsv1(x1)
        dsv = torch.cat([dsv5, dsv4, dsv3, dsv2], dim=1)
        dsv = self.scale_attention(dsv)
        # dsvv = torch.cat([dsv1,dsv],dim=1)
        layer_out = self.conv_out(dsv)
        final_out = decoder_out + layer_out
        return final_out, decoder_out, layer_out