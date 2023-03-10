import torch
import torchvision
import h5py
import copy
import math
# import numpy as np
# import tensorflow as tf
# import efficientnet.tfkeras as efn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from functools import reduce
from operator import __add__

# def replace_layers(model, old, new):
#     for n, module in model.named_children():
#         if len(list(module.children())) > 0:
#             ## compound module, go inside it
#             replace_layers(module, old, new)
#
#         if isinstance(module, old):
#             ## simple module
#             n = int(n)
#             model[n] = new

DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]


def load_pytorch_efficientnet(num_channels):
    model = torchvision.models.efficientnet_b0(weights='default')
    old = torch.nn.Conv2d
    new = torch.nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    replace_layers(model.features[0], old, new)
    return model


def round_filters(filters, divisor=8):
    """Round number of filters based on depth multiplier."""
    new_filters = max(
        divisor, int(filters + divisor / 2) // divisor * divisor
    )
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Args:
      input_shape: Input img shape
      kernel_size: An integer or tuple/list of 2 integers.
    Returns:
      A tuple.
    """
    img_dim = 2
    input_size = list(input_shape)[img_dim: (img_dim + 2)]
    # 128, 128
    if type(kernel_size):
        kernel_size = (kernel_size, kernel_size)
    # 3, 3
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        correct[1] - adjust[1], correct[1],
        correct[0] - adjust[0], correct[0],
    )


class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class EfficientNet(nn.Module):

    def __init__(self, input_shape, num_channels=5, dropout_rate=0.2, drop_connect_rate=0.2, include_top=True, pooling=None,
                 classes=1000, block_args = DEFAULT_BLOCKS_ARGS):

        super().__init__()
        block_args = copy.deepcopy(DEFAULT_BLOCKS_ARGS)
        rescale = lambda x: x * 1./255
        correct_padding =correct_pad(input_shape,kernel_size=3)
        # print(correct_padding, 'correct_padding')
        # self.input_transforms = transforms.Compose([
        #     transforms.Lambda(rescale),
        #     transforms.Normalize(0, 1),
        #     transforms.Lambda(nn.ZeroPad2d(padding=correct_padding)),
        #     ]
        # )

        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.pooling = pooling
        init_filters = round_filters(32)
        self.conv1 = Conv2dSamePadding(num_channels, init_filters, kernel_size=3, stride=2, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(init_filters, momentum=0.99, eps=0.001)
        self.swish = torch.nn.SiLU()
        b = 0
        blocks = float(sum(args['repeats'] for args in block_args))
        self.block_list = nn.ModuleList([])
        self.outputs = {}
        for i, args in enumerate(block_args):
            assert args['repeats'] > 0
            args["filters_in"] = round_filters(args["filters_in"])
            args["filters_out"] = round_filters(args["filters_out"])

            for j in range(args.pop("repeats")):
                if j > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]

                self.block_list.append(Block(drop_rate=drop_connect_rate * b / blocks, **args))
                b += 1
        output_filters = round_filters(1280)
        self.conv_final = Conv2dSamePadding(args["filters_out"], output_filters, kernel_size=1, stride=1,
                                    bias=False)
        self.batch_norm_final = nn.BatchNorm2d(output_filters, momentum=0.99, eps=0.001)

        if include_top:
            self.global_average_pooling_2d = lambda x: torch.mean(x, (2, 3), keepdim=False)
            if dropout_rate > 0:
                self.dropout_final = nn.Dropout(dropout_rate)
            self.linear = nn.Linear(output_filters, classes)
            self.softmax = nn.Softmax(dim=1)
        else:
            if pooling == 'avg':
                self.pool = lambda x: torch.mean(x, (2, 3), keepdim=False)
            elif pooling == 'max':
                self.pool = lambda x: torch.amax(x, (2, 3), keepdim=False)

    def forward(self, inputs):
        # padding = correct_pad(inputs, 3)
        # x = F.pad(input=inputs, pad=padding, value=0)
        # same as keras  x = layers.Rescaling(1.0 / 255.0)(x)

        # x = self.input_transforms(inputs)
        x = inputs
        self.outputs['pre_conv1'] = x.detach()
        # first conv layer
        x = self.conv1(x)
        self.outputs['conv1'] = x.detach()
        x = self.batch_norm1(x)
        self.outputs['bn1'] = x.detach()
        x = self.swish(x)
        self.outputs['activation1'] = x.detach()


        # go through all the conv blocks
        for idx, block_i in enumerate(self.block_list):
            x = block_i(x)
            self.outputs[f'block_{idx}'] = x.detach()

        # final conv layer
        x = self.conv_final(x)
        x = self.batch_norm_final(x)
        x = self.swish(x)
        self.outputs['final_conv_and_swish'] = x.detach()
        # specific outputs
        if self.include_top:
            x = self.global_average_pooling_2d(x)
            if self.dropout_rate > 0:
                x = self.dropout_final(x)
            x = self.linear(x)
            x = self.softmax(x)
        else:
            if self.pooling is not None:
                x = self.pool(x)
        return x


class Block(nn.Module):
    def __init__(self, drop_rate=0.0,
                 filters_in=32, filters_out=16,
                 kernel_size=3, strides=1,
                 expand_ratio=1, se_ratio=0.0,
                 id_skip=True, ):
        super().__init__()
        self.kernel_size = kernel_size
        filters = filters_in * expand_ratio
        self.expand_ratio = expand_ratio
        self.strides = strides
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.id_skip = id_skip
        self.drop_rate = drop_rate
        self.swish = torch.nn.SiLU()
        self.se_ratio = se_ratio
        self.outputs = {}
        if expand_ratio != 1:
            self.conv_expand = Conv2dSamePadding(filters_in, filters, kernel_size=1, bias=False)
            self.batch_norm_exp = nn.BatchNorm2d(filters,momentum=0.99, eps=0.001)

        # depth-wise convolution
        self.depth_conv = Conv2dSamePadding(filters, filters, kernel_size=self.kernel_size, stride=strides,
                                    bias=False, groups=filters)
        self.batch_norm_depth = nn.BatchNorm2d(filters,momentum=0.99, eps=0.001)

        # squeeze excitation phase
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            # implemented as https://github.com/keras-team/keras/blob/v2.11.0/keras/layers/pooling/global_average_pooling2d.py
            # assuming channels first because torch
            self.global_average_pooling_2d = lambda x: torch.mean(x, (2, 3), keepdim=True)
            self.conv_se_reduce = Conv2dSamePadding(filters, filters_se, 1, bias=True)
            self.sigmoid = torch.nn.Sigmoid()
            self.conv_se_expand = Conv2dSamePadding(filters_se, filters, 1, bias=True)
        self.conv_output = Conv2dSamePadding(filters, filters_out, 1, bias=False)
        self.batch_norm_last = nn.BatchNorm2d(filters_out,momentum=0.99, eps=0.001)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, inputs):
        x = inputs
        # if expanding the convolution
        if self.expand_ratio != 1:
            x = self.conv_expand(x)
            self.outputs['conv_expand'] = x.detach()
            x = self.batch_norm_exp(x)
            self.outputs['expand_bn'] = x.detach()
            x = self.swish(x)
            self.outputs['expand_activation'] = x.detach()
        # if need to account for stride
        # if self.strides == 2:
        #     padding = correct_pad(x.shape, self.kernel_size)
        #     out = F.pad(input=x, pad=padding, value=0)
        #     x = out

        x = self.depth_conv(x)
        self.outputs['depth_conv'] = x.detach()
        x = self.batch_norm_depth(x)
        self.outputs['depth_bn'] = x.detach()
        x = self.swish(x)
        self.outputs['depth_activation'] = x.detach()
        # squeeze and excitation
        if 0 < self.se_ratio <= 1:
            se = self.global_average_pooling_2d(x)
            se = self.conv_se_reduce(se)
            se = self.swish(se)
            se = self.conv_se_expand(se)
            se = self.sigmoid(se)
            x = torch.mul(x, se)
            self.outputs['post_se'] = x.detach()
        # output phase
        x = self.conv_output(x)
        x = self.batch_norm_last(x)
        if self.id_skip and self.strides == 1 and self.filters_in == self.filters_out:
            if self.drop_rate > 0:
                x = self.dropout(x)
            x = torch.add(x, inputs)

        return x


def EfficientNetB0(
        input_shape,
        include_top=True,
        pooling=None,
        classes=1000,
        block_args = DEFAULT_BLOCKS_ARGS,
        **kwargs,
):
    return EfficientNet(
        input_shape,
        dropout_rate=0.2,
        include_top=include_top,
        pooling=pooling,
        classes=classes,
        block_args = DEFAULT_BLOCKS_ARGS,
        **kwargs,
    )


if __name__ == '__main__':
    # model = load_pytorch_efficientnet(5)
    # rand_inp = torch.randn((1,5,256,256))
    # x = h5py.File('Cell_Painting_CNN_v1.hdf5','r')
    # print(model)
    #
    # def printname(name):
    #     print(name)

    # x.visit(printname)


    model = EfficientNetB0()

    inp = torch.randn((1,5,100,100))
    y = model(inp)
    print(y.shape)