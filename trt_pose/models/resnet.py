import torch
import torchvision
import torch.nn as nn
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


def _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        ResNetBackbone(resnet),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
    
    
def resnet18_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 512, num_upsample, num_flat)


def resnet34_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet34(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 512, num_upsample, num_flat)


def resnet50_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet50(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 2048, num_upsample, num_flat)


def resnet101_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet101(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 2048, num_upsample, num_flat)


def resnet152_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet152(pretrained=pretrained)
    return _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, 2048, num_upsample, num_flat)


def _resnet_pose(cmap_channels, paf_channels, upsample_channels, resnet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        ResNetBackbone(resnet),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
  
    
def _resnet_pose_att(cmap_channels, paf_channels, upsample_channels, resnet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        ResNetBackbone(resnet),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

    
def resnet18_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    return _resnet_pose_att(cmap_channels, paf_channels, upsample_channels, resnet, 512, num_upsample, num_flat)


def resnet34_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet34(pretrained=pretrained)
    return _resnet_pose_att(cmap_channels, paf_channels, upsample_channels, resnet, 512, num_upsample, num_flat)


def resnet50_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet50(pretrained=pretrained)
    return _resnet_pose_att(cmap_channels, paf_channels, upsample_channels, resnet, 2048, num_upsample, num_flat)


def resnet101_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet101(pretrained=pretrained)
    return _resnet_pose_att(cmap_channels, paf_channels, upsample_channels, resnet, 2048, num_upsample, num_flat)


def resnet152_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet152(pretrained=pretrained)
    return _resnet_pose_att(cmap_channels, paf_channels, upsample_channels, resnet, 2048, num_upsample, num_flat)


class ResNet18MultiscaleStride8(nn.Module):
    
    def __init__(self, cmap_channels, paf_channels, stride=8, num_expansion_channels=1024, num_mix_channels=128, num_mix_layers=1, pretrained_backbone=True):
        super(ResNet18MultiscaleStride8, self).__init__()
        
        num_cmap = cmap_channels
        num_paf = paf_channels
        
        self.backbone = torchvision.models.resnet18(pretrained=pretrained_backbone)
        
        self.up4 = nn.Sequential(
            CBR(512, 256),
            nn.UpsamplingNearest2d(scale_factor=2),
            CBR(256, 128),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.up3 = nn.Sequential(
            CBR(256, 128),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.att4 = GlobalAttention(512, 1024, act=nn.Tanh())
        self.att3 = GlobalAttention(256, 1024, act=nn.Tanh())
        
        mix = []
        mix += [ CBR(128*3, num_mix_channels, kernel_size=1) ]
        for i in range(num_mix_layers):
            mix += [ torchvision.models.resnet.BasicBlock(num_mix_channels, num_mix_channels) ]
        mix += [ CBR(num_mix_channels, num_expansion_channels)]
        self.mix = nn.Sequential(*mix)
        
        self.cmap = nn.Conv2d(num_expansion_channels, num_cmap, kernel_size=1, stride=1, padding=0)
        self.paf = nn.Conv2d(num_expansion_channels, num_paf, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x):
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x) # /4
        x2 = self.backbone.layer2(x1) # /8
        x3 = self.backbone.layer3(x2) # /16
        x4 = self.backbone.layer4(x3) # /32
        
        
        att = self.att4(x4) * self.att3(x3)
        
        x = torch.cat([self.up4(x4), self.up3(x3), x2], dim=1)
        x = self.mix(x) * att
        
        cmap = self.cmap(x)
        paf = self.paf(x)
        
        return cmap, paf
    
    
def resnet18_multiscale_stride8(*args, **kwargs):
    return ResNet18MultiscaleStride8(*args, **kwargs)