import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


class MaskConv(nn.Conv2d):
    def __init__(self, *args, is_reengineering, **kwargs):
        super(MaskConv, self).__init__(*args, **kwargs)

        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')

        self.is_reengineering = is_reengineering
        self.weight_mask, self.bias_mask = None, None
        if self.is_reengineering:
            self.init_mask()

    def init_mask(self):
        self.weight_mask = nn.Parameter(torch.ones(self.weight.size()))
        if self.bias is not None:
            self.bias_mask = nn.Parameter(torch.ones(self.bias.size()))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, inputs):
        if self.is_reengineering:
            bin_weight_mask = Binarization.apply(self.weight_mask)
            weight = self.weight * bin_weight_mask

            if self.bias is not None:
                bin_bias_mask = Binarization.apply(self.bias_mask)
                bias = self.bias * bin_bias_mask
            else:
                bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        output = F.conv2d(
            inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

        return output


class MaskLinear(nn.Linear):
    def __init__(self, *args, is_reengineering, **kwargs):
        super(MaskLinear, self).__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')

        self.is_reengineering = is_reengineering
        self.weight_mask, self.bias_mask = None, None
        if self.is_reengineering:
            self.init_mask()

    def init_mask(self):
        self.weight_mask = nn.Parameter(torch.ones(self.weight.size()))
        if self.bias is not None:
            self.bias_mask = nn.Parameter(torch.ones(self.bias.size()))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, inputs):
        if self.is_reengineering:
            bin_weight_mask = Binarization.apply(self.weight_mask)
            weight = self.weight * bin_weight_mask
            if self.bias is not None:
                bin_bias_mask = Binarization.apply(self.bias_mask)
                bias = self.bias * bin_bias_mask
            else:
                bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        output = F.linear(inputs, weight, bias)

        return output


class Binarization(autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        bin_mask = (mask > 0).float()
        return bin_mask

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)