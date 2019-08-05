import torch
import torchvision


class UpsampleCBR(torch.nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super(UpsampleCBR, self).__init__(
            torch.nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU()
        )
        
        
class PoseHead(torch.nn.Module):
    def __init__(self, input_channels, heatmap_channels, embedding_channels):
        super(PoseHead, self).__init__()
        self.heatmap_conv = torch.nn.Conv2d(input_channels, heatmap_channels, kernel_size=1, stride=1, padding=0)
        self.embedding_conv = torch.nn.Conv2d(input_channels, embedding_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        return self.heatmap_conv(x), self.embedding_conv(x)
    
    
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
    
    
def _resnet_pose(heatmap_channels, embedding_channels, upsample_channels, resnet, feature_channels):
    model = torch.nn.Sequential(
        ResNetBackbone(resnet),
        UpsampleCBR(feature_channels, upsample_channels),
        UpsampleCBR(upsample_channels, upsample_channels),
        UpsampleCBR(upsample_channels, upsample_channels),
        PoseHead(upsample_channels, heatmap_channels, embedding_channels)
    )
    return model


def resnet18_pose(heatmap_channels, embedding_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    return _resnet_pose(heatmap_channels, embedding_channels, upsample_channels, resnet, 512)


def resnet34_pose(heatmap_channels, embedding_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet34(pretrained=pretrained)
    return _resnet_pose(heatmap_channels, embedding_channels, upsample_channels, resnet, 512)


def resnet50_pose(heatmap_channels, embedding_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet50(pretrained=pretrained)
    return _resnet_pose(heatmap_channels, embedding_channels, upsample_channels, resnet, 2048)


def resnet101_pose(heatmap_channels, embedding_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet101(pretrained=pretrained)
    return _resnet_pose(heatmap_channels, embedding_channels, upsample_channels, resnet, 2048)


def resnet152_pose(heatmap_channels, embedding_channels, upsample_channels=256, pretrained=True):
    resnet = torchvision.models.resnet152(pretrained=pretrained)
    return _resnet_pose(heatmap_channels, embedding_channels, upsample_channels, resnet, 2048)