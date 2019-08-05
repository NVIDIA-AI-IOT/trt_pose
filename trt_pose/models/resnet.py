import torch
import torchvision
from .common import *


class ResNetBackbone(torch.nn.Module):
    
    def __init__(self, resnet):
        super(ResNetBackbone, self).__init__()
        self.resnet = resnet
    
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x) # /4
        x = self.resnet.layer2(x) # /8
        x = self.resnet.layer3(x) # /16
        x = self.resnet.layer4(x) # /32

        return x
    
    
def _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, feature_channels):
    model = torch.nn.Sequential(
        ResNetBackbone(resnet),
        UpsampleCBR(feature_channels, upsample_channels),
        UpsampleCBR(upsample_channels, upsample_channels),
        UpsampleCBR(upsample_channels, upsample_channels),
        PoseHead(upsample_channels, cmap_channels, paf_channels)
    )
    return model
    
    
def resnet18_pose(cmap_channels, paf_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 512)


def resnet34_pose(cmap_channels, paf_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 512)


def resnet50_pose(cmap_channels, paf_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 2048)


def resnet101_pose(cmap_channels, paf_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 2048)


def resnet152_pose(cmap_channels, paf_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 2048)

