import torch
import torchvision
from .common import *


class DenseNetBackbone(torch.nn.Module):
    
    def __init__(self, densenet):
        super(DenseNetBackbone, self).__init__()
        self.densenet = densenet
    
    def forward(self, x):
        x = self.densenet.features(x)
        return x
    
    
def _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        DenseNetBackbone(densenet),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
    
    
def _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        DenseNetBackbone(densenet),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

    
def densenet121_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet121(pretrained=pretrained)
    return _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, 1024, num_upsample, num_flat)


def densenet169_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet169(pretrained=pretrained)
    return _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, 1664, num_upsample, num_flat)


def densenet201_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet201(pretrained=pretrained)
    return _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, 1920, num_upsample, num_flat)


def densenet161_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet161(pretrained=pretrained)
    return _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, 2208, num_upsample, num_flat)


    
def densenet121_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet121(pretrained=pretrained)
    return _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, 1024, num_upsample, num_flat)


def densenet169_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet169(pretrained=pretrained)
    return _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, 1664, num_upsample, num_flat)


def densenet201_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet201(pretrained=pretrained)
    return _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, 1920, num_upsample, num_flat)


def densenet161_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet161(pretrained=pretrained)
    return _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, 2208, num_upsample, num_flat)


