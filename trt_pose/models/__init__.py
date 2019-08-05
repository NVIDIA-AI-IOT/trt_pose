from .resnet import *


MODELS = {
    'resnet18_u256': resnet18_pose,
    'resnet34_u256': resnet34_pose,
    'resnet50_u256': resnet50_pose,
    'resnet101_u256': resnet101_pose,
    'resnet152_u256': resnet152_pose
}