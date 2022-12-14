"""
Copy from https://github.com/chenyaofo/pytorch-cifar-models.
Then adding some codes for re-engineering.
"""

import sys
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Union, List, Dict, Any, cast
from binary_class.models.nn_layers import MaskConv, MaskLinear, Binarization


cifar10_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
}

cifar100_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        init_weights: bool = True,
        is_reengineering: bool = False
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            MaskLinear(512, 512, is_reengineering=is_reengineering),
            nn.ReLU(True),
            nn.Dropout(),
            MaskLinear(512, 512, is_reengineering=is_reengineering),
            nn.ReLU(True),
            nn.Dropout(),
            MaskLinear(512, num_classes, is_reengineering=is_reengineering),
        )

        self.is_reengineering = is_reengineering
        if is_reengineering:
            self.module_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_classes, 2)
            )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if hasattr(self, 'module_head'):
            x = self.module_head(x)

        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, MaskConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MaskLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, is_reengineering: bool=False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = MaskConv(in_channels, v, kernel_size=3, padding=1, is_reengineering=is_reengineering)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool,
         model_urls: Dict[str, str],
         pretrained: bool = False, progress: bool = True,
         is_reengineering: bool = False,
         **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, is_reengineering=is_reengineering),
                is_reengineering=is_reengineering, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)

        model_params = model.state_dict()
        model_params.update(state_dict)
        model.load_state_dict(model_params)
    return model


def cifar10_vgg11_bn(*args, **kwargs) -> VGG: pass
def cifar10_vgg13_bn(*args, **kwargs) -> VGG: pass
def cifar10_vgg16_bn(*args, **kwargs) -> VGG: pass
def cifar10_vgg19_bn(*args, **kwargs) -> VGG: pass


def cifar100_vgg11_bn(*args, **kwargs) -> VGG: pass
def cifar100_vgg13_bn(*args, **kwargs) -> VGG: pass
def cifar100_vgg16_bn(*args, **kwargs) -> VGG: pass
def cifar100_vgg19_bn(*args, **kwargs) -> VGG: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for cfg, model_name in zip(["A", "B", "D", "E"], ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_vgg,
                    arch=model_name,
                    cfg=cfg,
                    batch_norm=True,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )