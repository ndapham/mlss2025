import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Seg(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=True):
        super().__init__()
        # load torchvision's ResNet18
        resnet18 = models.resnet18(pretrained=pretrained)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            
            self.conv1.weight.data[:, :3] = resnet18.conv1.weight.data
            if in_channels > 3:
                
                nn.init.kaiming_normal_(self.conv1.weight.data[:, 3:], mode="fan_out", nonlinearity="relu")

        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        self.seg_head = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.seg_head(x)

        x = nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x