import torch
import torch.nn as nn
import torch.nn.functional as F

##from: https://code.likeagirl.io/u-net-vs-residual-u-net-vs-attention-u-net-vs-attention-residual-u-net-899b57c5698
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256, 128)

        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec5 = ConvBlock(128, 64)

        self.up6 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec6 = ConvBlock(64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        r1 = c1 + F.interpolate(p1, size=c1.shape[2:])  # residual add

        c2 = self.enc2(r1)
        p2 = self.pool2(c2)
        r2 = c2 + F.interpolate(p2, size=c2.shape[2:])

        c3 = self.enc3(r2)
        p3 = self.pool3(c3)
        r3 = c3 + F.interpolate(p3, size=c3.shape[2:])

        b = self.bottleneck(r3)

        u4 = self.up4(b)
        u4 = torch.cat([u4, c3], dim=1)
        c4 = self.dec4(u4)
        r4 = u4 + c4

        u5 = self.up5(r4)
        u5 = torch.cat([u5, c2], dim=1)
        c5 = self.dec5(u5)
        r5 = u5 + c5

        u6 = self.up6(r5)
        u6 = torch.cat([u6, c1], dim=1)
        c6 = self.dec6(u6)
        r6 = u6 + c6

        # output
        out = self.out_conv(r6)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out