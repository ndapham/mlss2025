import torch
import torch.nn as nn
import torch.nn.functional as F

##from: https://code.likeagirl.io/u-net-vs-residual-u-net-vs-attention-u-net-vs-attention-residual-u-net-899b57c5698

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        res = x if self.proj is None else self.proj(x)
        return out + res


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.theta_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi_g   = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.psi     = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.relu(self.theta_x(x))
        phi_g   = self.relu(self.phi_g(g))
        if phi_g.shape[2:] != theta_x.shape[2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode="bilinear", align_corners=False)
        f = theta_x + phi_g
        psi_f = self.sigmoid(self.psi(self.relu(f)))
        return x * psi_f

class ResAttentionUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()

        self.enc1 = ResidualConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ResidualConvBlock(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(in_channels=128, gating_channels=128, inter_channels=64)
        self.dec4 = ResidualConvBlock(256, 128)

        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att5 = AttentionBlock(in_channels=64, gating_channels=64, inter_channels=32)
        self.dec5 = ResidualConvBlock(128, 64)

        self.up6 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec6 = ResidualConvBlock(64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):

        c1 = self.enc1(x)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        c3 = self.enc3(p2)
        p3 = self.pool3(c3)

        b = self.bottleneck(p3)

        u4 = self.up4(b)
        c2_att = self.att4(c2, u4)
        u4 = torch.cat([u4, c2_att], dim=1)
        c4 = self.dec4(u4)

        u5 = self.up5(c4)
        c1_att = self.att5(c1, u5)
        u5 = torch.cat([u5, c1_att], dim=1)
        c5 = self.dec5(u5)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c1], dim=1)
        c6 = self.dec6(u6)

        out = self.out_conv(c6)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out