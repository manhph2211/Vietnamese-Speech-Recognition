import torch
from torch import nn
from utils.model_utils import get_same_padding

class GroupShuffle(nn.Module):

    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x


def get_conv_bn_layer(in_channels, out_channels, kernel_size=11,
                     stride=1, dilation=1, padding=0, bias=False,
                     groups=1, separable=False,
                     normalization="batch", norm_groups=1):
    if norm_groups == -1:
        norm_groups = out_channels

    if separable:
        layers = [
            nn.Conv1d(in_channels, in_channels, kernel_size,
                    stride=stride, dilation=dilation, padding=padding, bias=bias,
                    groups=in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1,
                    stride=1, dilation=1, padding=0, bias=bias, groups=groups)
        ]
    else:
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size,
                    stride=stride, dilation=dilation, padding=padding, bias=bias, groups=groups)
        ]

    if normalization == "group":
        layers.append(nn.GroupNorm(
            num_groups=norm_groups, num_channels=out_channels))
    elif normalization == "instance":
        layers.append(nn.GroupNorm(
            num_groups=out_channels, num_channels=out_channels))
    elif normalization == "layer":
        layers.append(nn.GroupNorm(
            num_groups=1, num_channels=out_channels))
    elif normalization == "batch":
        layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
    else:
        raise ValueError(
            f"Normalization method ({normalization}) does not match"
            f" one of [batch, layer, group, instance].")

    if groups > 1:
        layers.append(GroupShuffle(groups, out_channels))
    return nn.Sequential(*layers)


def get_act_dropout_layer(drop_prob=0.2, activation='relu'):
    if activation is None or activation == 'tanh':
        activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
    elif activation == 'relu':
        activation = nn.ReLU()
    layers = [
        activation,
        nn.Dropout(p=drop_prob)
    ]
    return nn.Sequential(*layers)


class MainBlock(nn.Module):
    def __init__(self, inplanes, planes, repeat=3, kernel_size=11, stride=1, residual=True,
             dilation=1, dropout=0.2, activation='relu',
             groups=1, separable=False, normalization="batch",
             norm_groups=1):
        super(MainBlock, self).__init__()
        padding_val = get_same_padding(kernel_size, stride, dilation)

        temp_planes = inplanes
        net = []
        for _ in range(repeat):
            net.append(
                get_conv_bn_layer(
                    temp_planes,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups)
            )
            net.append(
                get_act_dropout_layer(dropout, activation)
            )
            temp_planes = planes
        self.net = nn.Sequential(*net)
        self.residual = residual
        if self.residual:
            self.residual_layer = get_conv_bn_layer(
                                inplanes,
                                planes,
                                kernel_size=1,
                                normalization=normalization,
                                norm_groups=norm_groups)
        self.out = get_act_dropout_layer(dropout, activation)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            resudial = self.residual_layer(x)
            out += resudial
        return self.out(out)