import torch
import torchvision
from .common import *


class MnasnetBackbone(torch.nn.Module):
    
    def __init__(self, backbone):
        super(MnasnetBackbone, self).__init__()
        self.backbone = backbone
    
    def forward(self, x):
        x = self.backbone.layers(x)
        return x
    
    
def _mnasnet_pose(cmap_channels, paf_channels, upsample_channels, backbone, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        MnasnetBackbone(backbone),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
    
    
def _mnasnet_pose_att(cmap_channels, paf_channels, upsample_channels, backbone, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        MnasnetBackbone(backbone),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model


def mnasnet0_5_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    backbone = torchvision.models.mnasnet0_5(pretrained=pretrained)
    return _mnasnet_pose(cmap_channels, paf_channels, upsample_channels, backbone, 1280, num_upsample, num_flat)


def mnasnet0_75_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    backbone = torchvision.models.mnasnet0_75(pretrained=pretrained)
    return _mnasnet_pose(cmap_channels, paf_channels, upsample_channels, backbone, 1280, num_upsample, num_flat)


def mnasnet1_0_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    backbone = torchvision.models.mnasnet1_0(pretrained=pretrained)
    return _mnasnet_pose(cmap_channels, paf_channels, upsample_channels, backbone, 1280, num_upsample, num_flat)


def mnasnet1_3_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    backbone = torchvision.models.mnasnet1_3(pretrained=pretrained)
    return _mnasnet_pose(cmap_channels, paf_channels, upsample_channels, backbone, 1280, num_upsample, num_flat)



def mnasnet0_5_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    backbone = torchvision.models.mnasnet0_5(pretrained=pretrained)
    return _mnasnet_pose_att(cmap_channels, paf_channels, upsample_channels, backbone, 1280, num_upsample, num_flat)


def mnasnet0_75_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    backbone = torchvision.models.mnasnet0_75(pretrained=pretrained)
    return _mnasnet_pose_att(cmap_channels, paf_channels, upsample_channels, backbone, 1280, num_upsample, num_flat)


def mnasnet1_0_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    backbone = torchvision.models.mnasnet1_0(pretrained=pretrained)
    return _mnasnet_pose_att(cmap_channels, paf_channels, upsample_channels, backbone, 1280, num_upsample, num_flat)


def mnasnet1_3_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    backbone = torchvision.models.mnasnet1_3(pretrained=pretrained)
    return _mnasnet_pose_att(cmap_channels, paf_channels, upsample_channels, backbone, 1280, num_upsample, num_flat)