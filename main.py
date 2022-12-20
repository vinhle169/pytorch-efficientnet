import torch
import torchvision
import h5py
# import numpy as np
# import tensorflow as tf
# import efficientnet.tfkeras as efn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

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


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.
    Returns:
      A tuple.
    """
    img_dim = 2
    input_size = list(inputs.shape)[img_dim: (img_dim + 2)]
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
        correct[0] - adjust[0], correct[0],
        correct[1] - adjust[1], correct[1],
    )


class EfficientNet(nn.Module):

    def __init__(self, num_channels=3, dropout_rate=0.2, drop_connect_rate=0.2, include_top=True, pooling=None,
                 classes=1000):

        super().__init__()
        global DEFAULT_BLOCKS_ARGS
        self.transforms = torch.nn.Sequential(
            # transforms.to_tensor(),
            transforms.Normalize(0, 1)
        )
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.pooling = pooling
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding="valid", bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.swish = torch.nn.SiLU()
        b = 0
        blocks = float(sum(args['repeats'] for args in DEFAULT_BLOCKS_ARGS))
        self.block_list = nn.ModuleList([])
        for i, args in enumerate(DEFAULT_BLOCKS_ARGS):
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
        self.conv_final = nn.Conv2d(args["filters_out"], output_filters, kernel_size=1, stride=1, padding="same",
                                    bias=False)
        self.batch_norm_final = nn.BatchNorm2d(output_filters)

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
        padding = correct_pad(inputs, 3)
        x = F.pad(input=inputs, pad=padding, value=0)
        # first conv layer
        x = self.swish(self.batch_norm1(self.conv1(x)))

        # go through all the conv blocks
        for idx, block_i in enumerate(self.block_list):
            x = block_i(x)

        # final conv layer
        x = self.conv_final(x)
        x = self.batch_norm_final(x)
        x = self.swish(x)

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
        self.id_skip = id_skip
        self.swish = torch.nn.SiLU()
        self.se_ratio = se_ratio
        self.batch_norm_0 = nn.BatchNorm2d(filters)
        if expand_ratio != 1:
            self.conv_expand = nn.Conv2d(filters_in, filters, kernel_size=1, padding='same', bias=False)
            self.batch_norm_exp = nn.BatchNorm2d(filters)
        conv_pad = "valid" if self.strides == 2 else "same"
        self.depth_conv = nn.Conv2d(filters, filters * self.expand_ratio, kernel_size=self.kernel_size, stride=strides, padding=conv_pad,
                                    bias=False, groups=self.expand_ratio)
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            # implemented as https://github.com/keras-team/keras/blob/v2.11.0/keras/layers/pooling/global_average_pooling2d.py
            # assuming channels first because torch
            self.global_average_pooling_2d = lambda x: torch.mean(x, (2, 3), keepdim=True)
            self.conv_se_reduce = nn.Conv2d(filters, filters_se, 1, padding="same")
            self.sigmoid = torch.nn.Sigmoid()
            self.conv_se_expand = nn.Conv2d(filters_se, filters, 1, padding="same")
        self.conv_output = nn.Conv2d(filters, filters_out, 1, padding="same", bias=False)
        self.batch_norm_last = nn.BatchNorm2d(filters_out)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, inputs):
        x = inputs
        # if expanding the convolution
        if self.expand_ratio != 1:
            x = self.conv_expand(x)
            x = self.batch_norm_exp(x)
            x = self.swish(x)
        # if need to account for stride
        if self.stride == 2:
            padding = correct_pad(x, self.kernel_size)
            x = F.pad(input=x, pad=padding, value=0)
        x = self.depth_conv(x)
        x = self.batch_norm_0(x)
        x = self.swish(x)
        # squeeze and excitation
        if 0 < self.se_ratio <= 1:
            se = self.global_average_pooling_2d(x)
            se = self.conv_se_reduce(se)
            se = self.swish(se)
            se = self.conv_se_expand(se)
            se = self.sigmoid(se)
            x = torch.mul(x, se)
        # output phase
        x = self.conv_output(x)
        x = self.batch_norm_last(x)

        if self.id_skip and self.strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = self.dropout(x)
            x = torch.add(x, inputs)

        return x


def EfficientNetB0(
        include_top=True,
        pooling=None,
        classes=1000,
        **kwargs,
):
    return EfficientNet(
        dropout_rate=0.2,
        include_top=include_top,
        pooling=pooling,
        classes=classes,
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
