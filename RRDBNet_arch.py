import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Basic building block: a single convolutional layer ---
def make_layer(block, n_layers):
    """Stack n_layers copies of block into a Sequential module."""
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block with 5 convolution layers.
    Each layer takes ALL previous feature maps as input (dense connections).
    This is the core innovation of RDN/ESRGAN.
    """
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        # 5 conv layers; each gets progressively more input channels
        self.conv1 = nn.Conv2d(nf,        gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc,   gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        # Residual scaling: multiply by 0.2 to stabilize training
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB).
    3 RDB blocks stacked with another residual connection around them.
    """
    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x  # outer residual connection


class RRDBNet(nn.Module):
    """
    Full ESRGAN Generator Network.
    in_nc:  input channels (3 for RGB)
    out_nc: output channels (3 for RGB)
    nf:     number of feature maps (64)
    nb:     number of RRDB blocks (23)
    gc:     growth channels inside RDB (32)
    """
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super().__init__()
        RRDB_block = lambda: RRDB(nf, gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block, nb)      # 23 RRDB blocks
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # Upsampling layers (x4 total = two x2 steps using pixel shuffle)
        self.upconv1   = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2   = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv    = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk                               # global residual

        # Upsample x2, then x2 again → total x4
        fea = self.lrelu(self.upconv1(
            F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(
            F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out