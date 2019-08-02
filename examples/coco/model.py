import torch
import torchvision


class ResNetFeatureExtractor(torch.nn.Module):
    
    def __init__(self, resnet):
        super(ResNetFeatureExtractor, self).__init__()
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

        return [x1, x2, x3, x4]

    
class FeaturePyramid(torch.nn.Module):
    def __init__(self, feature_channels, output_channels, kernel_size=3):
        super(FeaturePyramid, self).__init__()
        self.num_feature_maps = len(feature_channels)
        self.output_channels = output_channels
        feature_convs = []
        for i in range(self.num_feature_maps):
            feature_convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(feature_channels[i], output_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
#                     torch.nn.BatchNorm2d(output_channels),
#                     torch.nn.ReLU()
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
                
        return xs

    
class Upsample(torch.nn.Module):
    
    def __init__(self, input_channels, output_channels, count):
        super(Upsample, self).__init__()
        layers = []
        for i in range(count):
            if i == 0:
                ch = input_channels
            else:
                ch = output_channels
            layers += [
                torch.nn.ConvTranspose2d(ch, output_channels, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU()
            ]
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    

class SelectInput(torch.nn.Module):
    
    def __init__(self, index):
        super(SelectInput, self).__init__()
        self.index = index
    
    def forward(self, inputs):
        return inputs[self.index]
    
    
class PoseHead(torch.nn.Module):
    def __init__(self, input_channels, cmap_channels, paf_channels):
        super(PoseHead, self).__init__()
        self.cmap_conv = torch.nn.Conv2d(input_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
        self.paf_conv = torch.nn.Conv2d(input_channels, paf_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        return self.cmap_conv(x), self.paf_conv(x)