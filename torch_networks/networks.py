import os, sys

sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.torch_loss_functions import angularLoss

import torchvision.models as models
import sys
import math
import pdb


# class ReLu6(nn.Module):
#     def __init__():
#         super(Relu6,self).__init__()
#     def forward(self,x):


class DilatedConvs(nn.Module):
    def __init__(self, in_ch, out_ch, num_dilation=4, dilate_rate=2, kernel_size=3):
        super(DilatedConvs, self).__init__()

        def ratio_of_featureMap(num_dilation, out_ch, dilate_rate=2):
            from math import ceil
            listA = [(i + 1) ** 1.5 for i in range(num_dilation)]
            sumA = sum(listA)
            # print listA
            return map(lambda x: int(ceil(out_ch * float(x) / float(sumA))), listA)[::-1]

        self.conv_layer_list = nn.ModuleList()
        self.in_ch = in_ch
        self.out_ch = out_ch
        same_padding = kernel_size // 2
        dilation = 1
        num_featureMap = ratio_of_featureMap(num_dilation, out_ch, dilate_rate=2)
        # print('n_feaure = {}'.format(num_featureMap))
        # dilation = []
        out_ch_concat = sum(num_featureMap)
        self.conv1x1_compress = nn.Conv2d(in_ch, out_ch // num_dilation, kernel_size=1, padding=0)
        self.conv1x1_decompress = nn.Conv2d(out_ch_concat, out_ch, kernel_size=1, padding=0)

        # same_padding =[1,2,4,8]
        for i in range(num_dilation):
            # padding = (kernel_size * dilation -1) //2
            # padding = same_padding[i]
            padding = dilation
            self.conv_layer_list.append(
                nn.Conv2d(out_ch // num_dilation, num_featureMap[i], kernel_size=kernel_size, dilation=dilation,
                          padding=padding))
            # self.conv_layer_list.append(nn.Conv2d(out_ch//num_dilation, out_ch//num_dilation, kernel_size=kernel_size, dilation=dilation, padding=padding))
            dilation = dilation * dilate_rate

    def forward(self, x):
        out_each = []
        # print('x shape = {} out_ch ={}, in_ch={}'.format(x.data[0].shape, self.out_ch, self.in_ch))
        x = self.conv1x1_compress(x)
        for conv in self.conv_layer_list:
            out_each.append(conv(x))
        out = torch.cat(out_each, 1)
        out = self.conv1x1_decompress(out)
        # print ('dilated outshape ={}'.format(out.data[0].shape))
        return out


class DownblockDilated(nn.Module):
    def __init__(self, in_ch, num_conv, ch_growth_rate, kernel_size=3):
        super(DownblockDilated, self).__init__()
        assert (num_conv > 0)
        self.in_ch = in_ch
        self.num_conv = num_conv
        self.ch_growth_rate = ch_growth_rate
        self.kernel_size = kernel_size
        self.layers = self.build_layer_block()
        self.block = nn.Sequential(*self.layers)
        # self.add_module('down_blocks',self.block)

    def forward(self, x):
        return self.block(x)

    def build_layer_block(self):
        layers = []
        same_padding = self.kernel_size // 2

        for i in range(self.num_conv):
            if i == 0:
                out_ch = self.in_ch * self.ch_growth_rate
            else:
                self.in_ch = out_ch
            layers.append(DilatedConvs(in_ch=self.in_ch, out_ch=out_ch))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU6())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False, ceil_mode=False))
        return layers


class Downblock(nn.Module):
    def __init__(self, in_ch, num_conv, ch_growth_rate, kernel_size=3):
        super(Downblock, self).__init__()
        assert (num_conv > 0)
        self.in_ch = in_ch
        self.num_conv = num_conv
        self.ch_growth_rate = ch_growth_rate
        self.kernel_size = kernel_size
        self.layers = self.build_layer_block()
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.block(x)

    def build_layer_block(self):
        layers = []
        same_padding = self.kernel_size // 2

        for i in range(self.num_conv):
            if i == 0:
                out_ch = self.in_ch * self.ch_growth_rate
            else:
                self.in_ch = out_ch
            layers.append(nn.Conv2d(self.in_ch, out_ch, kernel_size=self.kernel_size, padding=same_padding))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU6())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False, ceil_mode=False))
        return layers


class UpblockDilated(nn.Module):
    def __init__(self, in_ch, num_conv, ch_growth_rate, kernel_size=3):
        super(UpblockDilated, self).__init__()
        assert (num_conv > 0)
        self.in_ch = in_ch
        self.num_conv = num_conv
        self.ch_growth_rate = ch_growth_rate
        self.kernel_size = kernel_size
        self.layers = self.build_layer_block()
        self.block = nn.Sequential(*self.layers)
        # self.add_module('down_blocks',self.block)

    def forward(self, x):
        return self.block(x)

    def build_layer_block(self):
        layers = []
        same_padding = self.kernel_size // 2

        for i in range(self.num_conv):
            if i == 0:
                out_ch = self.in_ch // self.ch_growth_rate
            else:
                self.in_ch = out_ch
            layers.append(DilatedConvs(in_ch=self.in_ch, out_ch=out_ch))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU6())
        # layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False, ceil_mode=False))
        return layers


class Upblock(nn.Module):
    def __init__(self, in_ch, num_conv, ch_down_rate, kernel_size=3):
        super(Upblock, self).__init__()
        assert (num_conv > 0)
        self.in_ch = in_ch
        self.num_conv = num_conv
        self.ch_down_rate = ch_down_rate
        self.kernel_size = kernel_size
        self.layers = self.build_layer_block()
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.block(x)

    def build_layer_block(self):
        layers = []
        same_padding = self.kernel_size // 2
        print (self.num_conv)
        for i in range(self.num_conv):
            if i == 0:
                out_ch = self.in_ch // self.ch_down_rate
            else:
                self.in_ch = out_ch
            print(self.in_ch, self.kernel_size, same_padding)
            layers.append(nn.Conv2d(self.in_ch, out_ch, kernel_size=self.kernel_size, padding=same_padding))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU6())
        return layers


class conv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(conv_bn_relu, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.same_padding = (kernel_size - 1) / 2
        self.Conv = nn.Conv2d(self.in_ch, self.out_ch, self.same_padding)
        self.BatchNorm = nn.BatchNorm2d(out_ch)
        self.ReLU = nn.ReLU6(inplace=True)

    def forward(self, x):
        # x1 = self.Conv2d(x)
        # x1 = self.BatchNorm(x)
        # x1 = self.ReLU(x)
        return self.ReLU(self.BatchNorm(self.Conv(x)))


class Unet(nn.Module):
    def __init__(self, in_ch=1, first_out_ch=16, out_ch=1, number_bolck=4,
                 num_conv_in_block=2, ch_change_rate=2, kernel_size=3, target_label={'gradient': 2}):
        super(Unet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # self.ch_down_rate =ch_down_rate

        # self.conv_2d_1 = nn.Conv2d(in_ch, first_out_ch, kernel_size=kernel_size, padding=kernel_size //2 )

        self.conv_2d_1 = conv_bn_relu(in_ch, first_out_ch, kernel_size=kernel_size)

        self.down_block_1 = Downblock(first_out_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b2_down_ch = first_out_ch * ch_change_rate
        self.down_block_2 = Downblock(b2_down_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b3_down_ch = b2_down_ch * ch_change_rate
        self.down_block_3 = Downblock(b3_down_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b4_down_ch = b3_down_ch * ch_change_rate
        self.down_block_4 = Downblock(b4_down_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b5_down_ch = b4_down_ch * ch_change_rate
        self.down_block_5 = Downblock(b5_down_ch, num_conv_in_block, 1, kernel_size)
        # self.down_block_5=DownblockDilated(b5_down_ch,num_conv_in_block,1,kernel_size)
        # outout ch = 256

        b0_up_ch = b5_down_ch * 1

        # input ch = 512, output ch = 256
        self.up_block_0 = Upblock(b0_up_ch + b5_down_ch, num_conv_in_block, 2, kernel_size)
        b1_up_ch = (b0_up_ch + b5_down_ch) // 2
        self.up_block_1 = Upblock(b1_up_ch + b4_down_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b2_up_ch = (b1_up_ch + b4_down_ch) // ch_change_rate
        self.up_block_2 = Upblock(b2_up_ch + b3_down_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b3_up_ch = (b2_up_ch + b3_down_ch) // ch_change_rate
        self.up_block_3 = Upblock(b3_up_ch + b2_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        # self.up_block_3 = Upblock(160,num_conv_in_block,ch_change_rate,kernel_size)

        b4_up_ch = (b3_up_ch + b2_down_ch) // ch_change_rate
        # self.up_block_4 = Upblock(96,num_conv_in_block,ch_change_rate,kernel_size)
        self.up_block_4 = Upblock(b4_up_ch + first_out_ch, num_conv_in_block, ch_change_rate, kernel_size)

        last_up_ch = (b4_up_ch + first_out_ch) // ch_change_rate
        # self.finnal_conv2d = nn.Conv2d(last_up_ch, out_ch, kernel_size=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # if target_label:
        #     if 'gradient'in target_label:
        #         out_ch = target_label['gradient']
        #     elif 'distance' in target_label:
        #         out_ch = target_label['distance']
        self.finnal_conv2d = nn.Conv2d(last_up_ch, out_ch, kernel_size=3, padding=1)

    @property
    def name(self):
        return 'Unet'

    def forward(self, x):
        # x=self.finnal_conv2d(x)
        outputs = {}
        x1 = self.conv_2d_1(x)
        d_1 = self.down_block_1(x1)
        d_2 = self.down_block_2(d_1)
        d_3 = self.down_block_3(d_2)
        d_4 = self.down_block_4(d_3)
        d_5 = self.down_block_5(d_4)

        c_0 = torch.cat((self.upsample(d_5), d_4), 1)
        u_0 = self.up_block_0(c_0)

        c_1 = torch.cat((self.upsample(u_0), d_3), 1)
        u_1 = self.up_block_1(c_1)

        c_2 = torch.cat((self.upsample(u_1), d_2), 1)
        u_2 = self.up_block_2(c_2)

        c_3 = torch.cat((self.upsample(u_2), d_1), 1)
        u_3 = self.up_block_3(c_3)

        c_4 = torch.cat((self.upsample(u_3), x1), 1)
        u_4 = self.up_block_4(c_4)

        out = self.finnal_conv2d(u_4)
        return out


class _Unet_encoder_withDilatConv(nn.Module):
    def __init__(self, in_ch=1, first_out_ch=16, number_bolck=4, \
                 num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(_Unet_encoder_withDilatConv, self).__init__()
        self.in_ch = in_ch
        # self.out_ch = out_ch

        # self.conv_2d_1 = nn.Conv2d(in_ch, first_out_ch, kernel_size=kernel_size, padding=kernel_size // 2)

        self.conv_2d_1 = conv_bn_relu(in_ch, first_out_ch, kernel_size=kernel_size)

        self.enc_1 = Downblock(first_out_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b1_down_ch = first_out_ch * ch_change_rate

        self.enc_2 = Downblock(b1_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b2_down_ch = b1_down_ch * ch_change_rate

        self.enc_3 = Downblock(b2_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b3_down_ch = b2_down_ch * ch_change_rate

        # self.enc_4 = Downblock(b3_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        # print ('enc 4 has {}'.format(b3_down_ch))
        self.enc_4 = DownblockDilated(b3_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b4_down_ch = b3_down_ch * ch_change_rate

        # elf.enc_5 = Downblock(b4_down_ch, num_conv_in_block, 1, kernel_size)
        self.enc_5 = DownblockDilated(b4_down_ch, num_conv_in_block, 1, kernel_size)
        self.b5_down_ch = b4_down_ch * 1
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    @property
    def last_ch(self):
        return self.b5_down_ch

    def forward(self, x):
        x1 = self.conv_2d_1(x)
        d_1 = self.enc_1(x1)
        d_2 = self.enc_2(d_1)
        d_3 = self.enc_3(d_2)
        # print('d3 shape = {}'.format(d_3.data[0].shape))
        d_4 = self.enc_4(d_3)
        d_5 = self.enc_5(d_4)
        enc4_out = torch.cat((self.upsample(d_5), d_4), 1)
        return [enc4_out, d_3, d_2, d_1, x1]


class _Unet_encoder(nn.Module):
    def __init__(self, in_ch=1, first_out_ch=16, number_bolck=4, \
                 num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(_Unet_encoder, self).__init__()
        self.in_ch = in_ch
        # self.out_ch = out_ch

        # self.conv_2d_1 = nn.Conv2d(in_ch, first_out_ch, kernel_size=kernel_size, padding=kernel_size // 2)

        self.conv_2d_1 = conv_bn_relu(in_ch, first_out_ch, kernel_size=kernel_size)

        self.enc_1 = Downblock(first_out_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b1_down_ch = first_out_ch * ch_change_rate

        self.enc_2 = Downblock(b1_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b2_down_ch = b1_down_ch * ch_change_rate

        self.enc_3 = Downblock(b2_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b3_down_ch = b2_down_ch * ch_change_rate

        self.enc_4 = Downblock(b3_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b4_down_ch = b3_down_ch * ch_change_rate

        self.enc_5 = Downblock(b4_down_ch, num_conv_in_block, 1, kernel_size)
        # self.enc_5  =DownblockDilated(b4_down_ch, num_conv_in_block, 1, kernel_size)
        self.b5_down_ch = b4_down_ch * 1
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    @property
    def last_ch(self):
        return self.b5_down_ch

    def forward(self, x):
        x1 = self.conv_2d_1(x)
        d_1 = self.enc_1(x1)
        d_2 = self.enc_2(d_1)
        d_3 = self.enc_3(d_2)
        d_4 = self.enc_4(d_3)
        d_5 = self.enc_5(d_4)
        enc4_out = torch.cat((self.upsample(d_5), d_4), 1)
        return [enc4_out, d_3, d_2, d_1, x1]


class _Unet_decoder_withDilatConv(nn.Module):
    def __init__(self, bottom_input_ch, out_ch=2, num_conv_in_block=2, \
                 ch_change_rate=2, kernel_size=3, BatchNorm_final=True):
        super(_Unet_decoder_withDilatConv, self).__init__()
        b1_in_up_ch = bottom_input_ch + bottom_input_ch / 1
        # self.dec_1 = Upblock(b1_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)
        self.BatchNorm_final = BatchNorm_final
        self.dec_1 = UpblockDilated(b1_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b2_in_up_ch = b1_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 1)
        # self.dec_2 = Upblock(b2_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)
        self.dec_2 = UpblockDilated(b2_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b3_in_up_ch = b2_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 2)
        self.dec_3 = Upblock(b3_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b4_in_up_ch = b3_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 3)
        self.dec_4 = Upblock(b4_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b5_in_up_ch = b4_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 4)
        self.dec_5 = Upblock(b5_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        last_up_ch = b5_in_up_ch // ch_change_rate
        self.finnal_conv2d = nn.Conv2d(last_up_ch, out_ch, kernel_size=3, padding=(kernel_size - 1) / 2)
        self.BatchNorm = nn.BatchNorm2d(out_ch)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # return self.finnal_conv2d

    def forward(self, encoder_outputs):
        u_1 = self.dec_1(encoder_outputs[0])

        c_2 = torch.cat((self.upsample(u_1), encoder_outputs[1]), 1)
        u_2 = self.dec_2(c_2)

        c_3 = torch.cat((self.upsample(u_2), encoder_outputs[2]), 1)
        u_3 = self.dec_3(c_3)

        c_4 = torch.cat((self.upsample(u_3), encoder_outputs[3]), 1)
        u_4 = self.dec_4(c_4)

        c_5 = torch.cat((self.upsample(u_4), encoder_outputs[4]), 1)
        u_5 = self.dec_5(c_5)
        if self.BatchNorm_final:
            out = self.BatchNorm(self.finnal_conv2d(u_5))
        else:
            out = self.finnal_conv2d(u_5)
        return out


class _Unet_decoder(nn.Module):
    def __init__(self, bottom_input_ch, out_ch=2, num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(_Unet_decoder, self).__init__()
        b1_in_up_ch = bottom_input_ch + bottom_input_ch / 1
        self.dec_1 = Upblock(b1_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b2_in_up_ch = b1_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 1)
        self.dec_2 = Upblock(b2_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b3_in_up_ch = b2_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 2)
        self.dec_3 = Upblock(b3_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b4_in_up_ch = b3_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 3)
        self.dec_4 = Upblock(b4_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b5_in_up_ch = b4_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 4)
        self.dec_5 = Upblock(b5_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        last_up_ch = b5_in_up_ch // ch_change_rate
        self.finnal_conv2d = nn.Conv2d(last_up_ch, out_ch, kernel_size=3, padding=(kernel_size - 1) / 2)
        self.BatchNorm = nn.BatchNorm2d(out_ch)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # return self.finnal_conv2d

    def forward(self, encoder_outputs):
        u_1 = self.dec_1(encoder_outputs[0])

        c_2 = torch.cat((self.upsample(u_1), encoder_outputs[1]), 1)
        u_2 = self.dec_2(c_2)

        c_3 = torch.cat((self.upsample(u_2), encoder_outputs[2]), 1)
        u_3 = self.dec_3(c_3)

        c_4 = torch.cat((self.upsample(u_3), encoder_outputs[3]), 1)
        u_4 = self.dec_4(c_4)

        c_5 = torch.cat((self.upsample(u_4), encoder_outputs[4]), 1)
        u_5 = self.dec_5(c_5)

        out = self.BatchNorm(self.finnal_conv2d(u_5))
        return out


class MdecoderUnet_withDilatConv(nn.Module):
    def __init__(self, in_ch=1, first_out_ch=16, target_label={'nameless': 1},
                 number_bolck=4, num_conv_in_block=2, ch_change_rate=2, kernel_size=3, BatchNorm_final=True):
        super(MdecoderUnet_withDilatConv, self).__init__()
        self.in_ch = in_ch
        self.encoder = _Unet_encoder_withDilatConv(in_ch, first_out_ch, number_bolck, num_conv_in_block,
                                                   ch_change_rate, kernel_size)
        # self.add_module('encoder',self.encoder)
        self.target_label = target_label
        self.decoders = {}
        for name, out_ch in target_label.iteritems():
            self.decoders[name] = _Unet_decoder_withDilatConv(bottom_input_ch=self.encoder.last_ch,
                                                              out_ch=out_ch,
                                                              num_conv_in_block=num_conv_in_block,
                                                              BatchNorm_final=BatchNorm_final)
            self.add_module('decoder_' + name, self.decoders[name])

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        outputs = {}
        for name, decoder in self.decoders.iteritems():
            outputs[name] = decoder(encoder_outputs)
        return outputs

    @property
    def name(self):
        return 'MdecoderUnetDilatConv' + '_in_{}_chs'.format(self.in_ch)


class MaskMdecoderUnet_withDilatConv(MdecoderUnet_withDilatConv):
    def __init(self,**kwarg):
        super(MaskMdecoderUnet_withDilatConv,self).__init__(**kwarg)
    def forword(self,x):
        encoder_outputs = self.encoder(x)
        outputs = {}
        for name, decoder in self.decoders.iteritems():
            outputs[name] = torch.sigmoid(decoder(encoder_outputs))
        return outputs



class Mdecoder2Unet_withDilatConv(nn.Module):
    def __init__(self, mnet=None, freeze_net1=False, target_label={'unassigned': 1}, in_ch=3, out_ch=1, first_out_ch=16, \
                 number_bolck=4, num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(Mdecoder2Unet_withDilatConv, self).__init__()
        if not mnet:
            mnet = MdecoderUnet_withDilatConv(target_label=target_label, in_ch=in_ch)
        self.net1 = mnet
        total_input_ch = sum(self.net1.target_label.values()) + in_ch
        # print total_input_ch

        # print target_label
        net2_target_label = {}
        net2_target_label['final'] = out_ch
        self.net2 = MdecoderUnet_withDilatConv(target_label=net2_target_label, in_ch=total_input_ch,
                                               BatchNorm_final=False)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.add_module('first_' + self.net1.name, self.net1)
        self.add_module('second_' + self.net2.name, self.net2)

        # self.net2 = Unet()
        # #self.first_conv_in_net2 = conv_bn_relu(total_input_ch,first_out_ch)
        # self.first_conv_in_net2 = nn.Conv2d(total_input_ch,first_out_ch,kernel_size=kernel_size,padding =kernel_size // 2)
        # self.final_conv_in_net2 = nn.Conv2d(48,out_ch,kernel_size=kernel_size,padding =kernel_size // 2)
        # #self.BatchNorm = nn.BatchNorm2d(out_ch) 
        # self.net2.conv_2d_1     = self.first_conv_in_net2
        # self.net2.finnal_conv2d = self.final_conv_in_net2
        # self.out_ch = out_ch
        # self.in_ch  = in_ch
        # self.add_module('first_'+ self.net1.name, self.net1)
        # self.add_module('second_'+self.net2.name, self.net2)
        if freeze_net1:
            self.freezeWeight(self.net1)

    @property
    def name(self):
        return 'Mdecoder2Unet_withDilatConv' + '_in_{}_chs'.format(self.in_ch)

    def freezeWeight(self, net):
        for child in net.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.net1(x)
        out_chs = outputs.values()
        out_chs.append(x)
        x_net2_in = torch.cat(out_chs, 1)
        # outputs['final']   = self.net2(x_net2_in)
        # return outputs

        final_dict = self.net2(x_net2_in)
        outputs['final'] = final_dict[final_dict.keys()[0]]
        return outputs
        # return gradient_out,distance_out


class MdecoderUnet(nn.Module):
    def __init__(self, in_ch=1, first_out_ch=16, target_label={'nameless', 1}, \
                 number_bolck=4, num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(MdecoderUnet, self).__init__()
        self.in_ch = in_ch
        self.encoder = _Unet_encoder(in_ch, first_out_ch, number_bolck, num_conv_in_block, ch_change_rate, kernel_size)
        # self.add_module('encoder',self.encoder)
        self.target_label = target_label
        self.decoders = {}
        for name, out_ch in target_label.iteritems():
            self.decoders[name] = _Unet_decoder(bottom_input_ch=self.encoder.last_ch, out_ch=out_ch,
                                                num_conv_in_block=num_conv_in_block)
            self.add_module('decoder_' + name, self.decoders[name])

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        outputs = {}
        for name, decoder in self.decoders.iteritems():
            outputs[name] = decoder(encoder_outputs)
        return outputs

    @property
    def name(self):
        return 'MdecoderUnet' + '_in_{}_chs'.format(self.in_ch)


class Mdecoder2Unet(nn.Module):
    def __init__(self, mnet=None, freeze_net1=False, target_label={'unassigned': 1}, in_ch=1, out_ch=1, first_out_ch=16, \
                 number_bolck=4, num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(Mdecoder2Unet, self).__init__()
        if not mnet:
            mnet = MdecoderUnet(target_label=target_label, in_ch=in_ch)
        self.net1 = mnet
        total_input_ch = sum(self.net1.target_label.values()) + in_ch
        print total_input_ch
        self.net2 = Unet()
        # self.first_conv_in_net2 = conv_bn_relu(total_input_ch,first_out_ch)
        self.first_conv_in_net2 = nn.Conv2d(total_input_ch, first_out_ch, kernel_size=kernel_size,
                                            padding=kernel_size // 2)
        self.final_conv_in_net2 = nn.Conv2d(48, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.net2.conv_2d_1 = self.first_conv_in_net2
        self.net2.finnal_conv2d = self.final_conv_in_net2
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.add_module('first_' + self.net1.name, self.net1)
        self.add_module('second_' + self.net2.name, self.net2)
        if freeze_net1:
            self.freezeWeight(self.net1)

    @property
    def name(self):
        return 'Mdecoder2Unet' + '_in_{}_chs'.format(self.in_ch)

    def freezeWeight(self, net):
        for child in net.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x=self.finnal_conv2d(x)
        # outputs ={}
        outputs = self.net1(x)
        out_chs = outputs.values()
        out_chs.append(x)
        x_net2_in = torch.cat(out_chs, 1)
        # shape = x_net2.shape
        # x_net2_in  =   torch.nn.functional.normalize(out_chs.view(shape[0],shape[1],-1),2)
        # x_net2_in =  x_net2_in.view(shape)
        outputs['final'] = self.net2(x_net2_in)
        return outputs
        # return gradient_out,distance_out


class DUnet(nn.Module):
    def __init__(self, grad_unet=None, freeze_net1=True, in_ch=1, first_out_ch=16, out_ch=1, number_bolck=4, \
                 target_label=None, num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(DUnet, self).__init__()
        if grad_unet:
            self.net1 = grad_unet
        else:
            self.net1 = Unet(out_ch=2)
        self.net2 = Unet()
        self.first_conv_in_net2 = nn.Conv2d(3, 16, kernel_size=kernel_size, padding=kernel_size // 2)
        self.final_conv_in_net2 = nn.Conv2d(48, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.net2.conv_2d_1 = self.first_conv_in_net2
        self.net2.finnal_conv2d = self.final_conv_in_net2
        self.out_ch = out_ch
        if freeze_net1:
            self.freezeWeight(self.net1)

    @property
    def name(self):
        return 'DUnet_outch_{}'.format(self.out_ch)

    def freezeWeight(self, net):
        for child in net.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x=self.finnal_conv2d(x)
        outputs = {}
        outputs['gradient'] = self.net1(x)
        x_net2_in = torch.cat((outputs['gradient'], x), 1)
        outputs['final'] = self.net2(x_net2_in)
        return outputs
        # return gradient_out,distance_out


# class MaskUnet(MdecoderUnet_withDilatConv):
#    def __init__(self, **kwarg):
#        super(MaskUnet, self).__init__(**kwarg)



def test_angularLoss():
    A = torch.randn(16, 2, 224, 224).float()
    B = torch.randn(16, 2, 224, 224).float()
    B = A
    W = torch.randn(1, 1, 3, 3).float()
    # W = torch.abs(l2_norm(W))
    print(angularLoss(A, B, W))

def test_modules():
    MUnet = MdecoderUnet(target_label={'g': 2, 'a': 1, 'c': 2, 's': 1})
    print (MUnet)


if __name__ == '__main__':
    test_modules()
