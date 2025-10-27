import torch
import torch.nn as nn
class SKConv_7(nn.Module):
    """
    Selective Kernel Convolution (kernel size 7 variant)
    """
    def __init__(self, features, M=3, G=16, r=16, stride=1, L=32):
        """
        Args:
            features: input channel dimensionality.
            M: number of branches.
            G: number of convolution groups.
            r: reduction ratio for computing d.
            stride: stride of convolution.
            L: minimum dimension of the vector z.
        """
        super(SKConv_7, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=7,
                    stride=stride,
                    padding="same",
                    dilation=i + 1,
                    groups=G,
                    bias=False,
                ),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            )
            for i in range(M)
        ])

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )

        self.fcs = nn.ModuleList([
            nn.Conv2d(d, features, kernel_size=1, stride=1)
            for _ in range(M)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V


class SKConv(nn.Module):
    """
    Selective Kernel Convolution (kernel size 3 variant)
    """

    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        """
        Args:
            features: input channel dimensionality.
            M: number of branches.
            G: number of convolution groups.
            r: reduction ratio for computing d.
            stride: stride of convolution.
            L: minimum dimension of the vector z.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=3,
                    stride=stride,
                    padding="same",
                    dilation=i + 1,
                    groups=G,
                    bias=False,
                ),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            )
            for i in range(M)
        ])

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )

        self.fcs = nn.ModuleList([
            nn.Conv2d(d, features, kernel_size=1, stride=1)
            for _ in range(M)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V


class SKUnit(nn.Module):
    """
    Selective Kernel Residual Unit
    """
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=64):
        """
        Args:
            in_features: input channel dimensionality.
            mid_features: intermediate conv channels.
            out_features: output channel dimensionality.
            M: number of SK branches.
            G: num of conv groups.
            r: ratio for computing channel reduction.
            stride: stride.
            L: minimum vector dim (paper default = 32/64).
        """
        super(SKUnit, self).__init__()

        # 1x1 reduce
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
        )

        # SK convolution block
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)

        # 1x1 expand
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

        # Shortcut
        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        out += self.shortcut(residual)
        return self.relu(out)
