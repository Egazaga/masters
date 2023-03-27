import torch
from torch import nn


def calc_padding_size(kernel_size: int, padding=None):
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [s // 2 for s in kernel_size]
    return padding


class ConvolutionBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel: int = 1, stride: int = 1, groups: int = 1, padding=None,
                 use_bias=True):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_in,
                              out_channels=ch_out,
                              kernel_size=kernel,
                              stride=stride,
                              padding=calc_padding_size(kernel, padding),
                              groups=groups,
                              bias=use_bias)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class BottleneckBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, expansion: float = 0.5):
        super(BottleneckBlock, self).__init__()
        self.ch_hidden = int(ch_in * expansion)
        self.conv1 = ConvolutionBlock(ch_in, self.ch_hidden, kernel=1)
        self.conv2 = ConvolutionBlock(self.ch_hidden, ch_out, kernel=3)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out


class CspBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, bnecks: int, expansion: float = 0.5):
        super(CspBlock, self).__init__()
        self.ch_hidden = int(ch_in * expansion)
        self.conv1 = ConvolutionBlock(ch_in, self.ch_hidden, kernel=1)
        self.conv1_2 = ConvolutionBlock(self.ch_hidden, self.ch_hidden, kernel=1)
        self.conv2 = ConvolutionBlock(ch_in, self.ch_hidden, kernel=1)
        self.bneck = nn.Sequential(*[BottleneckBlock(self.ch_hidden, self.ch_hidden, 1.0) for _ in range(bnecks)])
        self.conv3 = ConvolutionBlock(2 * self.ch_hidden, ch_out, 1)

    def forward(self, x):
        x1 = self.conv1_2(self.bneck(self.conv1(x)))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))
