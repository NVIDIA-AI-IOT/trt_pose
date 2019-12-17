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

    

class _resnet18_custom_v0(nn.Module):
    
    def __init__(self, num_cmap, num_paf, upsample_channels=160, preconv_multiplier=4):
        super().__init__()
        self.num_cmap = num_cmap
        self.num_paf = num_paf
        
        self.resnet = torchvision.models.resnet18(pretrained=True)
        
        N = upsample_channels  # upsample features
        U = preconv_multiplier    # pre-conv expansion factor
        
        self.up4 = nn.Sequential(
            BlockUp(512, N),
            BlockUp(N, N),
            BlockUp(N, N)
        )
        
        self.global_att4 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(512, N * U, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(N * U, N * U, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.global_att3 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(256, N * U, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(N * U, N * U, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(128, N * U, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(N * U, N * U, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.pre_final = nn.Sequential(
            Block(N, N*U, kernel_size=1, stride=1),
        )
    
        self.final_cmap = nn.Conv2d(N*U, num_cmap, kernel_size=1, stride=1, padding=0)
        self.final_paf = nn.Conv2d(N*U, num_paf, kernel_size=1, stride=1, padding=0)
        
    
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x) # /4
        x2 = self.resnet.layer2(x1) # /8
        x3 = self.resnet.layer3(x2) # /16
        x4 = self.resnet.layer4(x3) # /32
        
        global_att = self.global_att4(x4) * self.global_att3(x3) * self.global_att2(x2)
        
        x = self.up4(x4)
        
        x = self.pre_final(x) * global_att
        
        cmap, paf = self.final_cmap(x), self.final_paf(x)
        
        return cmap, paf
    
    
def resnet18_custom_v0(cmap_channels, paf_channels, *args, **kwargs):
    return _resnet18_custom_v0(cmap_channels, paf_channels, *args, **kwargs)