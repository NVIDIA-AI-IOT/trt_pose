import torch
import torchvision


class FeatureExtractor(torch.nn.Module):
    
    def __init__(self, resnet):
        super(FeatureExtractor, self).__init__()
        self.resnet = resnet
    
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x) # /4
        x2 = self.resnet.layer2(x1) # /8
        x3 = self.resnet.layer3(x2) # /16
        x4 = self.resnet.layer4(x3) # /32

        return x1, x2, x3, x4

    
class FeaturePyramid(torch.nn.Module):
    def __init__(self, feature_channels, output_channels, kernel_size=1):
        super(FeaturePyramid, self).__init__()
        self.num_feature_maps = len(feature_channels)
        self.output_channels = output_channels
        feature_convs = []
        for i in range(self.num_feature_maps):
            feature_convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(feature_channels[i], output_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                    torch.nn.BatchNorm2d(output_channels),
                    torch.nn.ReLU()
                )
            ]
        self.feature_convs = torch.nn.ModuleList(feature_convs)
    
    def forward(self, *inputs):
        xs = []
        
        for i in range(self.num_feature_maps):
            cur_idx = - i - 1
            if i == 0:
                x = self.feature_convs[cur_idx](inputs[cur_idx])
            else:
                x += self.feature_convs[cur_idx](inputs[cur_idx])
            
            xs.insert(0, x)
                
            if i < self.num_feature_maps - 1:
                next_idx = - i - 2
                upsample_shape = inputs[next_idx].shape[2:]
                x = torch.nn.functional.interpolate(x, size=upsample_shape)
                
        return tuple(xs)

    
class PoseModel(torch.nn.Module):
    def __init__(self, feature_extractor, feature_pyramid, cmap_channels, paf_channels):
        super(PoseModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_pyramid = feature_pyramid
        self.cmap_conv = torch.nn.Conv2d(feature_pyramid.output_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
        self.paf_conv = torch.nn.Conv2d(feature_pyramid.output_channels, paf_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        xs = self.feature_extractor(x)
        xs = self.feature_pyramid(*xs)
        x = xs[0]
        return self.cmap_conv(x), self.paf_conv(x)

    
def baseline_resnet18(cmap_channels=18, paf_channels=38, fpn_channels=128, pretrained=False):
    
    resnet18 = torchvision.models.resnet18(pretrained=pretrained)
    feature_extractor = FeatureExtractor(resnet18)
    feature_pyramid = FeaturePyramid([64, 128, 256, 512], fpn_channels)
    model = PoseModel(feature_extractor, feature_pyramid, cmap_channels, paf_channels)
    
    return model