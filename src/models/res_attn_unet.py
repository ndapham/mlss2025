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
        return x2
    

class ResidualConvBlock(nn.Module):
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
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.enc1 = ResidualConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ResidualConvBlock(128, 256)

        self.up4 = Up(256, 128)
        self.att4 = AttentionBlock(in_channels=128, gating_channels=128, inter_channels=64)
        self.dec4 = ResidualConvBlock(256, 128)

        self.up5 = Up(128, 64)
        self.att5 = AttentionBlock(in_channels=64, gating_channels=64, inter_channels=32)
        self.dec5 = ResidualConvBlock(128, 64)

        self.up6 = Up(64, 32)
        self.att6 = AttentionBlock(in_channels=32, gating_channels=32, inter_channels=32)
        self.dec6 = ResidualConvBlock(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):

        c1 = self.enc1(x)   # (B, 32, W, H) 
        p1 = self.pool1(c1) # (B, 32, W/2, H/2)

        c2 = self.enc2(p1)  # (B, 64, W/2, H/2)
        p2 = self.pool2(c2) # (B, 64, W/4, H/4)

        c3 = self.enc3(p2)  # (B, 128, W/4, H/4)
        p3 = self.pool3(c3) # (B, 128, W/8, H/8)

        b = self.bottleneck(p3) # (B, 256, W/8, H/8)

        u4 = self.up4(c3, b)    # (B, 128, W/4, H/4)
        c3_att = self.att4(c3, u4)  # (B, 128, W/4, H/4)
        u4 = torch.cat([u4, c3_att], dim=1) # (B, 256, W/4, H/4)
        c4 = self.dec4(u4)  # (B, 128, W/4, H/4)

        u5 = self.up5(c2, c4)   # (B, 64, W/2, H/2)
        c2_att = self.att5(c2, u5) 
        u5 = torch.cat([u5, c2_att], dim=1)
        c5 = self.dec5(u5)

        u6 = self.up6(c1, c5)
        c1_att = self.att6(c1, u6) 
        u6 = torch.cat([u6, c1_att], dim=1)
        c6 = self.dec6(u6)

        out = self.out_conv(c6)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out