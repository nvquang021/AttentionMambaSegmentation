import torch
import torch.nn as nn
from einops import reduce
from src.models.modules.VSS_block import VSSBlock


# Priority Channel Attention (PCA)
class PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        # Compute the channel-wise mean of the input
        c = reduce(x, 'b c w h -> b c', 'mean')

        # Apply depthwise convolution
        x = self.dw(x)

        # Compute the channel-wise mean after convolution
        c_ = reduce(x, 'b c w h -> b c', 'mean')

        # Compute the attention scores
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))

        # Ensure dimensions match correctly for multiplication
        att_score = att_score.unsqueeze(2).unsqueeze(3)  # Shape [batch_size, channels, 1, 1]
        return x * att_score  # Broadcasting to match the dimensions
# Priority Spatial Attention (PSA)

class PSA(nn.Module):
    def __init__(self, dim,H=8,W=6):

        super().__init__()

        # self.dw = nn.Conv2d(dim, dim, kernel_size = 9, groups=dim, padding="same")
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.pw_h = nn.Conv2d(H, H, (1, 1))
        self.pw_w = nn.Conv2d(W, W, (1, 1))
        self.prob = nn.Softmax2d()
    def forward(self, x):

        s = reduce(x , 'b c w h -> b w h', 'mean')

        x1 = self.pw(x)
        x_h = self.pw_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.pw_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        s_ = reduce(x , 'b c w h -> b w h', 'mean')
        s_h = reduce(x_h , 'b c w h -> b w h', 'mean')
        s_w = reduce(x_w , 'b c w h -> b w h', 'mean')

        raise_sp = self.prob(s_ - s)
        raise_h = self.prob(s_h - s)
        raise_w = self.prob(s_w - s)

        att_score = torch.sigmoid(s_*(1 + raise_sp)+s_h*(1 + raise_h)+s_w*(1 + raise_w))
        att_score = att_score.unsqueeze(1)
        return x * att_score
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
class NewEncoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, H, W, drop_out = False):
    super(NewEncoderBlock, self).__init__()
    self.pca = PCA(in_channels//4)
    self.psa = PSA(in_channels//4, H, W)
    self.mamba = VSSBlock(hidden_dim = in_channels//4)
    self.axial1 = AxialMixer(in_channels//4, mixer_kernel=(3,3))
    self.axial2 = AxialMixer(in_channels//4, mixer_kernel=(3,3))
    self.resmamba = ResMambaBlock(in_channels//4)
    self.act = nn.ReLU()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.bn3 =  nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.ins_norm1 = nn.InstanceNorm2d(in_channels//4, affine=True)
    self.ins_norm2 = nn.InstanceNorm2d(in_channels//4, affine=True)
    self.ins_norm3 = nn.InstanceNorm2d(in_channels//4, affine=True)
    self.ins_norm4 = nn.InstanceNorm2d(in_channels//4, affine=True)
    self.down = nn.MaxPool2d((2,2))
    self.dropout = drop_out
  def forward(self, x):
    residual = x
    x1_,x2_,x3_,x4_ = torch.chunk(x,4, dim=1)
    x1 = self.mamba(self.pca(self.psa(x1_)).permute(0, 2, 3, 1))
    x1 = self.act(self.ins_norm1(x1.permute(0, 3, 1, 2) + x1_))
    x2 = self.mamba(self.pca(self.psa(x2_)).permute(0, 2, 3, 1))
    x2 = self.act(self.ins_norm1(x2.permute(0, 3, 1, 2) + x2_))
    x3 = self.axial1(self.resmamba(x3_))
    x3 = self.act(self.ins_norm1(x3 + x3_))
    x4 = self.axial2(self.resmamba(x4_))
    x4 = self.act(self.ins_norm1(x4 + x4_))
    x = torch.cat([x1, x2, x3, x4], dim=1)
    x = self.act(self.bn1(x))
    x = self.bn3(self.conv1(x))
    residual = self.bn2(self.conv2(residual))
    skip= x + residual
    x = self.down(skip)
    x = self.act(x)
    if self.dropout:
      x = nn.Dropout2d(0.3)(x)
    return x, skip
class ResMambaBlock(nn.Module):
    def __init__(self, in_c, k_size = 3):
      super().__init__()
      self.in_c = in_c
      self.conv = nn.Conv2d(in_c, in_c, k_size, stride=1, padding='same', dilation=1, groups=in_c, bias=True, padding_mode='zeros')
      self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
      self.act = nn.LeakyReLU(negative_slope=0.01)
      self.block = VSSBlock(hidden_dim = in_c)
      # self.block = nn.Conv2d(in_c, in_c, k_size, stride=1, padding='same')
      self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):

      skip = x

      x = self.conv(x)
      x = x.permute(0, 2, 3, 1)
      x = self.block(x)
      x = x.permute(0, 3, 1, 2)
      x = self.act(self.ins_norm(x))
      return x + skip * self.scale