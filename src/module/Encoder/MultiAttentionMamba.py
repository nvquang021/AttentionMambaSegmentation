
class MultiAttentionMamba(nn.Module):
  def __init__(self, in_channels, out_channels, H, W, drop_out = False):
    super(MultiAttentionMamba, self).__init__()
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
