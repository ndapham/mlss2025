import torch
import torch.nn as nn
import torch.nn.functional as F


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(128, 256)

        self.up4 = Up(256, 128)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec4 = ConvBlock(256, 128)

        self.up5 = Up(128, 64)
        self.conv5 = nn.Conv2d(128, 64,  kernel_size=3, padding=1)
        self.dec5 = ConvBlock(128, 64)

        self.up6 = Up(64, 32)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec6 = ConvBlock(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        
        c1 = self.enc1(x) #(B, 32, W, H)
        p1 = self.pool1(c1) #(B, 32, W/2, H/2)
        r1 = p1 + F.interpolate(c1, size=p1.shape[2:])  # residual add (B, 32, W/2, H/2)

        c2 = self.enc2(r1) #(B, 64, W, H)
        p2 = self.pool2(c2) #(B, 64, W/4, H/4)
        r2 = p2 + F.interpolate(c2, size=p2.shape[2:]) #(B, 64, W/4, H/4)
        
        c3 = self.enc3(r2)  #(B, 128, W, H)
        p3 = self.pool3(c3) #(B, 128, W/8, H/8)
        r3 = p3 + F.interpolate(c3, size=p3.shape[2:])  #(B, 128, W/8, H/8)
        
        b = self.bottleneck(r3) #(B, 256, W/8, H/8)

        u4 = self.up4(r3, b)    #(B, 256, W/8, H/8)
        c4 = self.conv4(u4)  # (B, 128, W/8, H/8 )
        r4 = c4 + self.dec4(u4)

        u5 = self.up5(r2, r4)
        c5 = self.conv5(u5)
        r5 = c5 + self.dec5(u5)

        u6 = self.up6(r1, r5)
        c6 = self.conv6(u6)
        r6 = c6 + self.dec6(u6)

        # output
        out = self.out_conv(r6)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out