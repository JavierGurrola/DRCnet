import torch
import torch.nn as nn
from collections import OrderedDict


class FactorizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(FactorizedBlock, self).__init__()

        self.conv_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=groups)),
            ('act', nn.PReLU(out_channels))
        ]))

        self.conv_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0), groups=groups)),
            ('act', nn.PReLU(out_channels))
        ]))

        self.conv_3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), groups=groups)),
            ('act', nn.PReLU(out_channels))
        ]))

        self.conv_4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(3 * out_channels, out_channels, kernel_size=(1, 1, 1), groups=groups)),
            ('act', nn.PReLU(out_channels))
        ]))

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x)
        x_3 = self.conv_3(x)

        x = torch.cat([x_1, x_2, x_3], 1)
        x = self.conv_4(x)

        return x


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, inner_convolutions=1, residual=False, groups=16):
        super(DenoisingBlock, self).__init__()

        self.inner_convolutions = inner_convolutions
        self.residual = residual

        self.input_convolution = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, inner_channels, kernel_size=(1, 1, 1))),
            ('act', nn.PReLU(inner_channels))
        ]))

        self.dense_convolutions = nn.ModuleList()
        for i in range(1, inner_convolutions + 1):
            dense_convolution = FactorizedBlock(in_channels + i * inner_channels, inner_channels, groups)
            self.dense_convolutions.append(dense_convolution)

        self.output_convolution = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels=in_channels + (inner_convolutions + 1) * inner_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 1, 1))),
            ('act', nn.PReLU(out_channels))
        ]))

    def forward(self, x):
        output = self.input_convolution(x)
        output = torch.cat([x, output], 1)

        for i in range(self.inner_convolutions):
            inner_output = self.dense_convolutions[i](output)
            output = torch.cat([output, inner_output], 1)

        output = self.output_convolution(output)

        if self.residual:
            output += x

        return output


class InputBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(InputBlock, self).__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.out_channels = out_channels

        self.input_conv = nn.Sequential(OrderedDict([
            ('conv 0', nn.Conv3d(in_channels, inner_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('act 0', nn.PReLU(inner_channels)),
            ('conv 1', nn.Conv3d(in_channels, inner_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('act 1', nn.PReLU(inner_channels))
        ]))

    def forward(self, inputs):
        inputs = self.input_conv(inputs)        # 1 x M x N
        down = self.down_conv(inputs)           # K x M/2 x N/2

        return down, inputs


class OutputBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv_t = nn.ConvTranspose3d(in_channels, inner_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.act_t = nn.PReLU(inner_channels)

        self.output_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(inner_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('act', nn.PReLU(out_channels))
        ]))

    def forward(self, x, residual):
        x = self.act_t(self.conv_t(x)) + residual
        return self.output_conv(x)


class GatedBlock(nn.Module):
    def __init__(self, x_channels, h_channels, dense_convs=1, groups=1):
        super(GatedBlock, self).__init__()
        self._test = False

        self.z_t = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(h_channels + x_channels, h_channels, kernel_size=(1, 1, 1))),
            ('act', nn.Sigmoid())
        ]))

        self.r_t = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(h_channels + x_channels, h_channels, kernel_size=(1, 1, 1))),
            ('act', nn.Sigmoid())
        ]))

        self.h_t = DenoisingBlock(h_channels + x_channels, h_channels, h_channels, dense_convs, False, groups)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.size(), dtype=x.dtype, layout=x.layout, device=x.device)

        concat = torch.cat([h, x], dim=1)
        z_t = self.z_t(concat)
        r_t = self.r_t(concat)
        concat = torch.cat([r_t * h, x], dim=1)
        h_t = self.h_t(concat)

        h_t = (1 - z_t) * h + z_t * h_t

        if self._test:
            return h_t, z_t
        else:
            return h_t


class DenoiserNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        channels = kwargs['input channels']
        groups = kwargs['groups']
        self._test = False

        dense_convs = kwargs['dense convolutions']
        self.residual = kwargs['residual']
        filters_0 = kwargs['base filters']
        filters_1 = filters_0

        self.input_block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(channels, filters_0, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('act', nn.PReLU(filters_0))
        ]))

        self.down_block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(filters_0, filters_1, kernel_size=(2, 2, 2), stride=(2, 2, 2))),
            ('act', nn.PReLU(filters_1)),
        ]))

        self.up_block = nn.Sequential(OrderedDict([
            ('conv', nn.ConvTranspose3d(filters_1, filters_0, kernel_size=(2, 2, 2), stride=(2, 2, 2))),
            ('act', nn.PReLU(filters_0)),
        ]))

        self.denoising_block = GatedBlock(filters_1, filters_1, dense_convs, groups)

        self.output_block = nn.Sequential(OrderedDict([
            ('conv 0', nn.Conv3d(2 * filters_0, filters_0, kernel_size=(1, 1, 1))),
            ('act 0', nn.PReLU(filters_0)),
            ('conv 1', nn.Conv3d(filters_0, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('act 1', nn.PReLU(channels))
        ]))

    def forward(self, inputs):
        up_0 = self.input_block(inputs)
        x = self.down_block(up_0)

        # x 1
        h = None
        if self._test:
            z_list = []
        for i in range(4):

            if self._test:
                h, z = self.denoising_block(x, h)
                z_list.append(z)
            else:
                h = self.denoising_block(x, h)
            x += h

        up_3 = self.up_block(x)

        if self._test:
            return self.output_block(torch.cat([up_0, up_3], 1)) + inputs, z_list

        return self.output_block(torch.cat([up_0, up_3], 1)) + inputs

    def test(self, flag):
        self._test = flag
        self.denoising_block._test = flag
