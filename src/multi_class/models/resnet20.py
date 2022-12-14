"""
Copy from https://github.com/chenyaofo/pytorch-cifar-models.
Then adding some codes for re-engineering.
"""
import sys
import torch.nn as nn
import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional
from models.nn_layers import MaskConv, MaskLinear, Binarization


cifar10_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
}

cifar100_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
}


def conv3x3(in_planes, out_planes, stride=1, is_reengineering=False):
    """3x3 convolution with padding"""
    return MaskConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, is_reengineering=is_reengineering)


def conv1x1(in_planes, out_planes, stride=1, is_reengineering=False):
    """1x1 convolution"""
    return MaskConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, is_reengineering=is_reengineering)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_reengineering=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, is_reengineering=is_reengineering)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, is_reengineering=is_reengineering)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, is_reengineering=False, num_classes_in_super: int = -1,):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16, is_reengineering=is_reengineering)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], is_reengineering=is_reengineering)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, is_reengineering=is_reengineering)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, is_reengineering=is_reengineering)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MaskLinear(64 * block.expansion, num_classes, is_reengineering=is_reengineering)


        self.is_reengineering = is_reengineering
        self.num_classes_in_super = num_classes_in_super
        if is_reengineering:
            assert num_classes_in_super > 0
            self.module_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_classes, num_classes_in_super)
            )

        for m in self.modules():
            if isinstance(m, MaskConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, is_reengineering=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, is_reengineering=is_reengineering),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, is_reengineering=is_reengineering))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_reengineering=is_reengineering))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if hasattr(self, 'module_head'):
            x = self.module_head(x)

        return x


    def get_masks(self):
        masks = {k: v for k, v in self.state_dict().items() if 'mask' in k}
        return masks

    def count_weight_ratio(self):
        masks = []
        for n, layer in self.named_modules():
            if hasattr(layer, 'weight_mask'):
                masks.append(torch.flatten(layer.weight_mask))
                if layer.bias_mask is not None:
                    masks.append(torch.flatten(layer.bias_mask))

        masks = torch.cat(masks, dim=0)
        bin_masks = Binarization.apply(masks)
        weight_ratio = torch.mean(bin_masks)
        return weight_ratio

    def get_module_head(self):
        head = {k: v for k, v in self.state_dict().items() if 'module_head' in k}
        return head


def _resnet(
    arch: str,
    layers: List[int],
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    is_reengineering: bool = False,
    num_classes_in_super: int = -1,
    **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(BasicBlock, layers, is_reengineering=is_reengineering, num_classes_in_super=num_classes_in_super, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_params = model.state_dict()
        model_params.update(state_dict)
        model.load_state_dict(model_params)
    return model


def cifar10_resnet20(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet32(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet44(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet56(*args, **kwargs) -> CifarResNet: pass


def cifar100_resnet20(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet32(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet44(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet56(*args, **kwargs) -> CifarResNet: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for layers, model_name in zip([[3]*3, [5]*3, [7]*3, [9]*3],
                                  ["resnet20", "resnet32", "resnet44", "resnet56"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_resnet,
                    arch=model_name,
                    layers=layers,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )