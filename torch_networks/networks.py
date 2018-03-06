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
from clstm  import ConvLSTMCell

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
        self.conv_out_ch = 0
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
        self.conv_out_ch = out_ch
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
        self.conv_out_ch =0
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
        self.conv_out_ch = out_ch
        # layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False, ceil_mode=False))
        return layers

class UpblockDialted_CLSTM(UpblockDilated):
    def __init_(self,in_ch, num_conv, ch_growth_rate, kernel_size=3):
        super(UpblockDilated_CLSTM, self).__init__(in_ch_num_conv, ch_growth_rate, kernel_size)
        self.clstm = ConvLSTMCell(self.conv_out_ch, 
                                  hidden_size = self.conv_out_ch, 
                                  kernel_size = kernel_size, 
                                  padding = 1)
        self.prev_stat = None
    def forward(self,x):
        x = super().forward(x)
        self.prev_stat = self.clstm(x, self.prev_stat)
        x, cell = self.prev_stat
        return x

class DownblockDilated_CLSTM(DownblockDilated):
    def __init__(self, in_ch, num_conv, ch_growth_rate, kernel_size=3):
        super(DownblockDilated_CLSTM, self).__init__(in_ch,num_conv,ch_growth_rate,kernel_size)
        self.clstm = ConvLSTMCell(self.conv_out_ch, 
                                  hidden_size = self.conv_out_ch, 
                                  kernel_size = kernel_size, 
                                  padding = 1)
        self.prev_state = None

        # assert (num_conv > 0)
        # self.in_ch = in_ch
        # self.num_conv = num_conv
        # self.ch_growth_rate = ch_growth_rate
        # self.kernel_size = kernel_size
        # self.layers = self.build_layer_block()
        # self.block = nn.Sequential(*self.layers)
        # self.add_module('down_blocks',self.block)

    def forward(self, x):
        #return self.block(x)
        x=super().forward(x)
        self.prev_stat = self.clstm(x, self.prev_stat)
        x, cell = self.prev_stat
        return x

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
        #print (self.num_conv)
        for i in range(self.num_conv):
            if i == 0:
                out_ch = self.in_ch // self.ch_down_rate
            else:
                self.in_ch = out_ch
            #print(self.in_ch, self.kernel_size, same_padding)
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
        self.first_out_ch =first_out_ch
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


class _Unet_encoder_withFullDilatConv(nn.Module):
    def __init__(self, in_ch=1, first_out_ch=16, number_bolck=4, \
                 num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(_Unet_encoder_withFullDilatConv, self).__init__()
        self.in_ch = in_ch
        self.first_out_ch =first_out_ch
        # self.out_ch = out_ch

        # self.conv_2d_1 = nn.Conv2d(in_ch, first_out_ch, kernel_size=kernel_size, padding=kernel_size // 2)

        self.conv_2d_1 = conv_bn_relu(in_ch, first_out_ch, kernel_size=kernel_size)

        self.enc_1 = DownblockDilated(first_out_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b1_down_ch = first_out_ch * ch_change_rate

        self.enc_2 = DownblockDilated(b1_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b2_down_ch = b1_down_ch * ch_change_rate

        self.enc_3 = DownblockDilated(b2_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
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

class _Unet_decoder_withFullDilatConv(nn.Module):
    def __init__(self, bottom_input_ch, out_ch=2, num_conv_in_block=2, \
                 ch_change_rate=2, kernel_size=3, BatchNorm_final=True):
        super(_Unet_decoder_withFullDilatConv, self).__init__()
        b1_in_up_ch = bottom_input_ch + bottom_input_ch / 1
        # self.dec_1 = Upblock(b1_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)
        self.BatchNorm_final = BatchNorm_final
        self.dec_1 = UpblockDilated(b1_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b2_in_up_ch = b1_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 1)
        # self.dec_2 = Upblock(b2_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)
        self.dec_2 = UpblockDilated(b2_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b3_in_up_ch = b2_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 2)
        self.dec_3 =UpblockDilated(b3_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b4_in_up_ch = b3_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 3)
        self.dec_4 = UpblockDilated(b4_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b5_in_up_ch = b4_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 4)
        self.dec_5 = UpblockDilated(b5_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

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



class _Unet_multi_head_centerGated_decoder(nn.Module):
    def __init__(self, bottom_input_ch,
                  tg_name_and_ch={'nameless': 1},
                  number_bolck=4, num_conv_in_block=2, 
                  ch_change_rate=2, kernel_size=3, 
                  BatchNorm_final=True,
                  compressed_gates = False):
        super(_Unet_multi_head_centerGated_decoder,self).__init__()
        self.num_branchs = len(tg_name_and_ch)
        self.tg_name_and_ch = tg_name_and_ch
        self.BatchNorm_final = BatchNorm_final

       # print('num_branchs = {}'.format(self.num_branchs))
        b1_in_up_ch = bottom_input_ch + bottom_input_ch
        self.dec_1 =  nn.ModuleList([UpblockDilated(b1_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)
                       for i in range(self.num_branchs)])
        b2_in_up_ch = b1_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 1)
        self.dec_2 =  nn.ModuleList([UpblockDilated(b2_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)
                      for i in range(self.num_branchs)])

        b3_in_up_ch = b2_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 2)
        self.dec_3 =  nn.ModuleList([Upblock(b3_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)
                       for i in range(self.num_branchs)])

        b4_in_up_ch = b3_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 3)
        self.dec_4 =  nn.ModuleList([Upblock(b4_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)
                       for i in range(self.num_branchs)])

        b5_in_up_ch = b4_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 4)
        self.dec_5 =  nn.ModuleList([Upblock(b5_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)
                       for i in range(self.num_branchs)])


        last_up_ch = b5_in_up_ch // ch_change_rate
        self.finnal_conv2d =  nn.ModuleList([nn.Conv2d(last_up_ch, out_ch, kernel_size=3, padding=(kernel_size - 1) / 2)
                              for name, out_ch in tg_name_and_ch.iteritems()]
                              )

        

        if compressed_gates:
            self.CG1  = _centerGated_compressed_(b1_in_up_ch // ch_change_rate, self.num_branchs)
            self.CG2  = _centerGated_compressed_(b2_in_up_ch // ch_change_rate, self.num_branchs)
            self.CG3  = _centerGated_compressed_(b3_in_up_ch // ch_change_rate, self.num_branchs)
            self.CG4  = _centerGated_compressed_(b4_in_up_ch // ch_change_rate, self.num_branchs)
            self.CG5  = _centerGated_compressed_(last_up_ch, self.num_branchs)
        else:
            self.CG1  = _centerGated_output_(b1_in_up_ch // ch_change_rate, self.num_branchs)
            self.CG2  = _centerGated_output_(b2_in_up_ch // ch_change_rate, self.num_branchs)
            self.CG3  = _centerGated_output_(b3_in_up_ch // ch_change_rate, self.num_branchs)
            self.CG4  = _centerGated_output_(b4_in_up_ch // ch_change_rate, self.num_branchs)
            self.CG5  = _centerGated_output_(last_up_ch, self.num_branchs)

        

        self.BatchNorm = nn.ModuleList([nn.BatchNorm2d(out_ch) for name, out_ch in tg_name_and_ch.iteritems()])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def set_multi_gpus(self):
        # self.dec_1 = nn.DataParallel(self.dec_1).cuda()
        # self.dec_2 = nn.DataParallel(self.dec_2).cuda()
        # self.dec_3 = nn.DataParallel(self.dec_3).cuda()
        # self.dec_4 = nn.DataParallel(self.dec_4).cuda()
        # self.dec_5 = nn.DataParallel(self.dec_5).cuda()


        self.dec_1 = nn.ModuleList([ nn.DataParallel(module).cuda() for module in self.dec_1])
        self.dec_2 = nn.ModuleList([ nn.DataParallel(module).cuda() for module in self.dec_2])
        self.dec_3 = nn.ModuleList([ nn.DataParallel(module).cuda() for module in self.dec_3])
        self.dec_4 = nn.ModuleList([ nn.DataParallel(module).cuda() for module in self.dec_4])
        self.dec_5 = nn.ModuleList([ nn.DataParallel(module).cuda() for module in self.dec_5])

        self.CG1 = nn.DataParallel(self.CG1).cuda()
        self.CG2 = nn.DataParallel(self.CG2).cuda()
        self.CG3 = nn.DataParallel(self.CG3).cuda()
        self.CG4 = nn.DataParallel(self.CG4).cuda()
        self.CG5 = nn.DataParallel(self.CG5).cuda()
        self.BatchNorm = nn.ModuleList([ nn.DataParallel(module).cuda() for module in self.BatchNorm])
        #self.BatchNorm = nn.DataParallel(self.BatchNorm).cuda()
        self.upsample = nn.DataParallel(self.upsample).cuda()
        self.finnal_conv2d = nn.ModuleList([ nn.DataParallel(module).cuda() for module in self.finnal_conv2d])
        #self.finnal_conv2d =nn.DataParallel(self.finnal_conv2d).cuda()


    def forward(self, encoder_outputs):
        u1     = [dec(encoder_outputs[0]) for dec in self.dec_1]
        u1_igate ,u1_fgate = self.CG1(u1)
        u1     = [ (u * f_gate)+i_gate  for u, (i_gate,f_gate) in zip(u1,zip(u1_igate ,u1_fgate))]

        c2     =  [torch.cat((self.upsample(u), encoder_outputs[1]), 1) for u in u1]
        u2     =  [dec(c) for dec, c in zip(self.dec_2,c2)]
        u2_igate ,u2_fgate = self.CG2(u2)
        u2     = [ (u * f_gate)+i_gate  for u, (i_gate,f_gate) in zip(u2,zip(u2_igate ,u2_fgate))]

            
        c3     =  [torch.cat((self.upsample(u), encoder_outputs[2]), 1) for u in u2]
        u3     =  [dec(c) for dec, c in zip(self.dec_3,c3)]
        u3_igate ,u3_fgate = self.CG3(u3)
        u3     = [ (u * f_gate)+i_gate  for u, (i_gate,f_gate) in zip(u3,zip(u3_igate ,u3_fgate))]


        c4     =  [torch.cat((self.upsample(u), encoder_outputs[3]), 1) for u in u3]
        u4     =  [dec(c) for dec, c in zip(self.dec_4,c4)]
        u4_igate ,u4_fgate = self.CG4(u4)
        u4     = [ (u * f_gate)+i_gate  for u, (i_gate,f_gate) in zip(u4,zip(u4_igate ,u4_fgate))]

            
        c5     =  [torch.cat((self.upsample(u), encoder_outputs[4]), 1) for u in u4 ]
        u5     =  [dec(c) for dec, c in zip(self.dec_5,c5)]
        u5_igate ,u5_fgate = self.CG5(u5)
        u5     = [ (u * f_gate)+i_gate  for u, (i_gate,f_gate) in zip(u5,zip(u5_igate ,u5_fgate))]


        out = [conv(u) for conv,u in zip(self.finnal_conv2d, u5)]
        if self.BatchNorm_final:
               out = [bn(conv(u)) for bn,(conv,u) in zip(self.BatchNorm,zip(self.finnal_conv2d, u5))]
        else:
               out = [conv(u) for conv,u in zip(self.finnal_conv2d, u5)]

        out_dict ={}
        iters = 0
        for name, ch in self.tg_name_and_ch.iteritems():
            out_dict[name] = out[iters]
            iters += 1
        return out_dict

        # u1     = [dec(encoder_outputs[0]) for dec in self.dec_1]
        # u1_res = self.CG1(u1)
        # u1     = [ u + u_res for u, u_res in zip(u1,u1_res)]

        # c2     =  [torch.cat((self.upsample(u), encoder_outputs[1]), 1) for u in u1]
        # u2     =  [dec(c) for dec, c in zip(self.dec_2,c2)]
        # u2_res =  self.CG2(u2)
        # u2     = [ u + u_res for u, u_res in zip(u2, u2_res)]

            
        # c3     =  [torch.cat((self.upsample(u), encoder_outputs[2]), 1) for u in u2]
        # u3     =  [dec(c) for dec, c in zip(self.dec_3,c3)]
        # u3_res =  self.CG3(u3)
        # u3     = [ u + u_res for u, u_res in zip(u3,u3_res)]


        # c4     =  [torch.cat((self.upsample(u), encoder_outputs[3]), 1) for u in u3]
        # u4     =  [dec(c) for dec, c in zip(self.dec_4,c4)]
        # u4_res =  self.CG4(u4)
        # u4     = [ u + u_res for u , u_res in zip(u4, u4_res)]

            
        # c5     =  [torch.cat((self.upsample(u), encoder_outputs[4]), 1) for u in u4 ]
        # u5     =  [dec(c) for dec, c in zip(self.dec_5,c5)]
        # u5_res =  self.CG5(u5)
        # u5     = [ u + u_res for u,u_res in zip(u5, u5_res)]


        # out = [conv(u) for conv,u in zip(self.finnal_conv2d, u5)]
        # if self.BatchNorm_final:
        #        out = [bn(conv(u)) for bn,(conv,u) in zip(self.BatchNorm,zip(self.finnal_conv2d, u5))]
        # else:
        #        out = [conv(u) for conv,u in zip(self.finnal_conv2d, u5)]

        # out_dict ={}
        # iters = 0
        # for name, ch in self.tg_name_and_ch.iteritems():
        #     out_dict[name] = out[iters]
        #     iters += 1
        # return out_dict


class _centerGated_output_(nn.Module):
    def __init__(self,input_chs, num_branchs):
        super(_centerGated_output_ , self).__init__()
        self.input_chs   = input_chs
        self.num_branchs = num_branchs
        self.Gates = nn.Conv2d(input_chs * num_branchs, input_chs * num_branchs, kernel_size=3, padding=1)
        self.ConvOuts = nn.Conv2d(input_chs * num_branchs, input_chs * num_branchs, kernel_size=3, padding=1)
        self.batchNorm =  nn.BatchNorm2d(input_chs * num_branchs)

    def forward(self, tensorlist):
        #print('centerGate tensor shape = {}'.format(tensorlist[0].shape))
        tensorStack = torch.cat(tensorlist,1)
        out_gates = torch.sigmoid(self.Gates(tensorStack))
        #gates_list = out_gates.chunk(self.num_branchs,1)
        conv_out  = self.ConvOuts(tensorStack)
        conv_out  = self.batchNorm(conv_out)
        out = out_gates * conv_out
        out_tuple = out.chunk(self.num_branchs,1)
        return out_tuple

class _centerGated_compressed_(nn.Module):
    def __init__(self,input_chs, num_branchs):
        super(_centerGated_compressed_, self).__init__()
        self.input_chs   = input_chs
        self.num_branchs = num_branchs
       
        self.batchNorm =  nn.BatchNorm2d(input_chs * num_branchs)

        self.compress_Conv = nn.Conv2d(input_chs * num_branchs, input_chs, kernel_size=1, padding=0)
        self.in_Gates = nn.Conv2d(input_chs, input_chs, kernel_size=3, padding=1)
        self.forget_Gates = nn.Conv2d(input_chs, input_chs, kernel_size=3, padding=1)
        self.ConvOuts = nn.Conv2d(input_chs , input_chs , kernel_size=3, padding=1)
        self.decompress_Conv = nn.Conv2d(input_chs, input_chs * num_branchs, kernel_size=1, padding=0)
        self.decompress_convGate = nn.Conv2d(input_chs, input_chs * num_branchs, kernel_size=1, padding=0)



        
        #self.Gates = nn.Conv2d(input_chs * num_branchs, input_chs * num_branchs, kernel_size=3, padding=1)
        #self.ConvOuts = nn.Conv2d(input_chs * num_branchs, input_chs * num_branchs, kernel_size=3, padding=1)
        self.batchNorm1 =  nn.BatchNorm2d(input_chs)
        self.batchNorm2 =  nn.BatchNorm2d(input_chs*num_branchs)
        self.ReLU = nn.ReLU6(inplace=True)

    def forward(self, tensorlist):
        #print('centerGate tensor shape = {}'.format(tensorlist[0].shape))
        #
        tensorStack = torch.cat(tensorlist,1)
        compressed_stack = self.compress_Conv(tensorStack)
        in_gates = torch.sigmoid(self.in_Gates(compressed_stack))
        forget_gates = torch.sigmoid(self.in_Gates(compressed_stack))
        #gates_list = out_gates.chunk(self.num_branchs,1)
        conv_out  = torch.tanh(self.ConvOuts(compressed_stack))
        #conv_out  = self.ReLU(self.batchNorm1(conv_out))
        in_conv = in_gates * conv_out
        f_gates = torch.sigmoid(self.forget_Gates(compressed_stack))
        decompressed_out = self.decompress_Conv(in_conv)
        decompressed_fgates=self.decompress_convGate(f_gates)
        #decompressed_out  = self.ReLU(self.batchNorm2(decompressed_out))
        in_gate_tuple = decompressed_out.chunk(self.num_branchs,1)
        f_gate_tuple  = decompressed_fgates.chunk(self.num_branchs,1)
        return in_gate_tuple, f_gate_tuple


class _Unet_multi_head_centerGatedCLSTM_decoder(_Unet_multi_head_centerGated_decoder):
    def __init__(self, bottom_input_ch,
                  tg_name_and_ch={'nameless': 1},
                  number_bolck=4, num_conv_in_block=2, 
                  ch_change_rate=2, kernel_size=3, 
                  BatchNorm_final=True):
        super(_Unet_multi_head_centerGatedCLSTM_decoder,self).__init__(bottom_input_ch,
                                                                    tg_name_and_ch,
                                                                    number_bolck, num_conv_in_block, 
                                                                    ch_change_rate, kernel_size, 
                                                                    BatchNorm_final)
        self.CG1  = _centerGate_withCLSTM_(b1_in_up_ch // ch_change_rate, self.num_branchs)
        self.CG2  = _centerGate_withCLSTM_(b2_in_up_ch // ch_change_rate, self.num_branchs)
        self.CG3  = _centerGate_withCLSTM_(b3_in_up_ch // ch_change_rate, self.num_branchs)
        self.CG4  = _centerGate_withCLSTM_(b4_in_up_ch // ch_change_rate, self.num_branchs)
        self.CG5  = _centerGate_withCLSTM_(last_up_ch, self.num_branchs)

    def reset_clstm_state(self):
        self.CG1.reset_clstm_state()
        self.CG2.reset_clstm_state()
        self.CG3.reset_clstm_state()
        self.CG4.reset_clstm_state()
        self.CG5.reset_clstm_state()

        
    def forward(self, encoder_outputs):
        return super().forward(encoder_outputs)

class _centerGate_withCLSTM_(nn.Module):
    def __init__(self,input_chs, num_branchs):
        super(_centerGated_output_ , self).__init__()
        self.input_chs   = input_chs
        self.num_branchs = num_branchs
        self.compress_Conv = nn.Conv2d(input_chs * num_branchs, input_chs, kernel_size=1, padding=0)
        self.Gates_clstm = ConvLSTMCell(input_chs, input_chs, kernel_size =3, padding=1, use_gpu =True)
        self.ConvOuts_clstm = ConvLSTMCell(input_chs, input_chs, kernel_size =3, padding=1, use_gpu =True)
        self.decompress_Conv = nn.Conv2d(input_chs, input_chs * num_branchs, kernel_size=1, padding=0)
        
        #self.Gates = nn.Conv2d(input_chs * num_branchs, input_chs * num_branchs, kernel_size=3, padding=1)
        #self.ConvOuts = nn.Conv2d(input_chs * num_branchs, input_chs * num_branchs, kernel_size=3, padding=1)
        self.batchNorm =  nn.BatchNorm2d(input_chs * num_branchs)
        self.ReLU = nn.ReLU6(inplace=True)
        self.prev_stat = None

    def reset_clstm_state(self):
        self.prev_stat = None

    def forward(self, tensorlist):
        #print('centerGate tensor shape = {}'.format(tensorlist[0].shape))
        tensorStack = torch.cat(tensorlist,1)
        compressed_stack = self.compress_Conv(tensorStack)
        out_gates = torch.sigmoid( self.Gates_clstm(compressed_stack))
        prev_stat  = self.ConvOuts_clstm(compressed_stack, prev_stat)
        #conv_out  = prev_stat[0]
        conv_out  = self.batchNorm(prev_stat[0])
        out = out_gates * conv_out
        decompressed_out = self.decompress_Conv(out)
        out_tuple = decompressed_out.chunk(self.num_branchs,1)
        return out_tuple


class _Unet_decoder_withDilatCLSTM(nn.Module):
    def __init__(self, bottom_input_ch, out_ch=2, num_conv_in_block=2, \
                 ch_change_rate=2, kernel_size=3, BatchNorm_final=True):
        super(_Unet_decoder_withDilatConv, self).__init__()
        b1_in_up_ch = bottom_input_ch + bottom_input_ch / 1
        # self.dec_1 = Upblock(b1_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)
        self.BatchNorm_final = BatchNorm_final
        self.dec_1 = UpblockDialted_CLSTM(b1_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b2_in_up_ch = b1_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 1)
        # self.dec_2 = Upblock(b2_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)
        self.dec_2 = UpblockDialted_CLSTM(b2_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b3_in_up_ch = b2_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 2)
        self.dec_3 = UpblockDialted_CLSTM(b3_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b4_in_up_ch = b3_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 3)
        self.dec_4 = UpblockDialted_CLSTM(b4_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

        b5_in_up_ch = b4_in_up_ch // ch_change_rate + bottom_input_ch // (2 ** 4)
        self.dec_5 = UpblockDialted_CLSTM(b5_in_up_ch, num_conv_in_block, ch_change_rate, kernel_size)

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
        self.first_out_ch = first_out_ch
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


class MdecoderUnet_withFullDilatConv(nn.Module):
    def __init__(self, in_ch=1, first_out_ch=16, target_label={'nameless': 1},
                 number_bolck=4, num_conv_in_block=2, ch_change_rate=2, kernel_size=3, BatchNorm_final=True):
        super(MdecoderUnet_withFullDilatConv, self).__init__()
        self.in_ch = in_ch
        self.first_out_ch = first_out_ch
        self.encoder = _Unet_encoder_withFullDilatConv(in_ch, first_out_ch, number_bolck, num_conv_in_block,
                                                   ch_change_rate, kernel_size)
        # self.add_module('encoder',self.encoder)
        self.target_label = target_label
        self.decoders = {}
        for name, out_ch in target_label.iteritems():
            self.decoders[name] = _Unet_decoder_withFullDilatConv(bottom_input_ch=self.encoder.last_ch,
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
        return 'MdecoderUnetFullDilatConv' + '_in_{}_chs'.format(self.in_ch)


class MdecoderUnet_withDilatConv_centerGateCLSTM(nn.Module):
    def __init__(self, in_ch=1, first_out_ch=16, target_label={'nameless': 1},
                 number_bolck=4, num_conv_in_block=2, ch_change_rate=2, kernel_size=3, BatchNorm_final=True):
        super(MdecoderUnet_withDilatConv_centerGateCLSTM, self).__init__()
        self.in_ch = in_ch
        self.first_out_ch = first_out_ch
        self.encoder = _Unet_encoder_withDilatConv(1, first_out_ch, number_bolck, num_conv_in_block,
                                                   ch_change_rate, kernel_size)
        # self.add_module('encoder',self.encoder)
        self.target_label = target_label
        self.decoder = _Unet_multi_head_centerGatedCLSTM_decoder(bottom_input_ch=self.encoder.last_ch,
                                                                tg_name_and_ch=self.target_label,
                                                                BatchNorm_final=BatchNorm_final)
        self.add_module('decoder',self.decoder)

    def forward(self, x):
        self.decoder.reset_clstm_state()
        chs = x.size(1)
        x_in_chs = x.chunk(chs,1)
        #for x_ch in x_in_chs:
        encoder_outputs = [self.encoder(x_ch) for x_ch in x_in_chs]
        decoder_outputs = [self.decoder(encoder_out) for encoder_out in encoder_outputs]
        #outputs = self.decoder(encoder_outputs)
        outputs = torch.cat(decoder_outputs,1)
        return outputs

    @property
    def name(self):
        return 'MdecoderUnetDilatConvCenterGateCLSTM' + '_in_{}_chs'.format(self.in_ch)


class MdecoderUnet_withDilatConv_centerGate(nn.Module):
    def __init__(self, in_ch=1, first_out_ch=16, target_label={'nameless': 1},
                 number_bolck=4, num_conv_in_block=2, ch_change_rate=2, kernel_size=3, BatchNorm_final=True):
        super(MdecoderUnet_withDilatConv_centerGate, self).__init__()
        self.in_ch = in_ch
        self.first_out_ch = first_out_ch
        self.encoder = _Unet_encoder_withDilatConv(in_ch, first_out_ch, number_bolck, num_conv_in_block,
                                                   ch_change_rate, kernel_size)
        # self.add_module('encoder',self.encoder)
        self.target_label = target_label
        self.use_compressed_gates = True
        self.decoder = _Unet_multi_head_centerGated_decoder(bottom_input_ch=self.encoder.last_ch,
                                                             tg_name_and_ch=self.target_label,
                                                             BatchNorm_final=BatchNorm_final,
                                                             compressed_gates = self.use_compressed_gates)
        self.add_module('decoder',self.decoder)
    def set_multi_gpus(self):
        self.encoder = nn.DataParallel(self.encoder).cuda()
        
        self.decoder.set_multi_gpus()

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        outputs = self.decoder(encoder_outputs)
        return outputs

    @property
    def name(self):
        gate_str = '_centerCompressedGate' if self.use_compressed_gates else '_centerGate'
        return 'MdecoderUnetDilat' +  gate_str + '_in_{}_chs'.format(self.in_ch)






class MaskMdecoderUnet_withDilatConv(MdecoderUnet_withDilatConv):
    def __init(self,**kwarg):
        super(MaskMdecoderUnet_withDilatConv,self).__init__(**kwarg)
    def forword(self,x):
        encoder_outputs = self.encoder(x)
        outputs = {}
        for name, decoder in self.decoders.iteritems():
            outputs[name] = torch.sigmoid(decoder(encoder_outputs))
        return outputs
    
    @property
    def name(self):
        #print('name in first ch {}'.formart(self.first_out_ch))
        return 'MaskMdecoderDilatUnet' \
               + '-In_{}_CHS'.format(self.in_ch) \
               +'-FisrtConv_{}_CHS'.format(self.first_out_ch)


class Mdecoder2Unet_withDilatConv(nn.Module):
    def __init__(self, mnet=None, freeze_net1=False, target_label={'unassigned': 1}, net2_target_label=None,
                 label_catin_net2=None, in_ch=3, out_ch=1, first_out_ch=16, \
                 number_bolck=4, num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(Mdecoder2Unet_withDilatConv, self).__init__()
        self.label_catin_net2 =label_catin_net2
        self.target_label = target_label
        if not mnet:
            mnet = MdecoderUnet_withDilatConv(target_label=target_label, in_ch=in_ch)
        self.net1 = mnet

        if not label_catin_net2:
            total_input_ch = sum(self.net1.target_label.values()) + in_ch
        else:
            total_input_ch = sum([self.net1.target_label[lb] for lb in label_catin_net2]) +in_ch

        if not net2_target_label:
            net2_target_label = {}
            net2_target_label['final'] = out_ch
        self.net2 = MdecoderUnet_withDilatConv(target_label=net2_target_label, in_ch=total_input_ch,
                                               BatchNorm_final=False)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.add_module('first_' + self.net1.name, self.net1)
        self.add_module('second_' + self.net2.name, self.net2)

        if freeze_net1:
            self.freezeWeight(self.net1)


    @property
    def name(self):
        return 'Mdecoder2Unet_in_{}_chs_'.format(self.in_ch) + \
        '_['+ '-'.join(self.label_catin_net2 if self.label_catin_net2 else self.target_label.keys())
        # label_ch_list = self.label_catin_net2 if self.label_catin_net2 else self.target_label.keys()
        # cat_in_lable = '_['+'_'.joint(label_ch_list)+']'
        # return 'Mdecoder2Unet_in_{}_chs_'.format(self.in_ch) + cat_in_lable

    def freezeWeight(self, net):
        for child in net.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.net1(x)
        #out_chs = outputs.values()
        
        if not self.label_catin_net2:
            out_chs = outputs.values()
            #print('.  cat all')
        else:
            #print('.   cat some')
            out_chs = [outputs[lb] for lb in self.label_catin_net2]

        out_chs.append(x)
        x_net2_in = torch.cat(out_chs, 1)
        final_dict = self.net2(x_net2_in)
        #outputs['final'] = final_dict[final_dict.keys()[0]]
        outputs.update(final_dict)
        return outputs








class Mdecoder2Unet_withDilatConv_LSTM_on_singleOBJ(nn.Module):
    def __init__(self, mnet=None, freeze_net1=False, target_label={'unassigned': 1}, label_catin_net2=None, in_ch=3, out_ch=1, first_out_ch=16, \
                 number_bolck=4, num_conv_in_block=2, ch_change_rate=2, kernel_size=3):
        super(Mdecoder2Unet_withDilatConv_LSTM_on_singleOBJ, self).__init__()
        self.label_catin_net2 =label_catin_net2
        self.target_label = target_label
        if not mnet:
            mnet = MdecoderUnet_withDilatConv(target_label=target_label, in_ch=in_ch)
        self.net1 = mnet

        if not label_catin_net2:
            total_input_ch = sum(self.net1.target_label.values()) + in_ch
        else:
            total_input_ch = sum([self.net1.target_label[lb] for lb in label_catin_net2]) +in_ch

        net2_target_label = {}
        net2_target_label['final'] = out_ch
        print ('final outout ch = {}'.format(out_ch))
        self.net2 = MdecoderUnet_withDilatConv(target_label=net2_target_label, in_ch=total_input_ch,
                                               BatchNorm_final=True)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.add_module('first_' + self.net1.name, self.net1)
        self.add_module('second_' + self.net2.name, self.net2)
        self.max_loop_size  =30
        self.clstm = ConvLSTMCell(24, 24, kernel_size =3,padding=1, use_gpu =True)
        self.ReLU = nn.ReLU6(inplace=True)
        #self.clstm2 = ConvLSTMCell(16, 16, kernel_size =3,padding=1, use_gpu =True)
        self.compress_Conv = nn.Conv2d(24, 1, 1)

        if freeze_net1:
            self.freezeWeight(self.net1)


    @property
    def name(self):
        return 'Mdecoder2Unet_in_{}_chs_'.format(self.in_ch) + \
        '_['+ '-'.join(self.label_catin_net2 if self.label_catin_net2 else self.target_label.keys()) +\
        '_CLSTM'

    def freezeWeight(self, net):
        for child in net.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.net1(x)
        #out_chs = outputs.values()
        
        if not self.label_catin_net2:
            out_chs = outputs.values()
            #print('.  cat all')
        else:
            #print('.   cat some')
            out_chs = [outputs[lb] for lb in self.label_catin_net2]

        out_chs.append(x)
        #ch_shape =out_chs[0].shape
        # out_chs =[functional.normalize(
        #                                ch.view(-1,ch.shape[2]*ch.shape[3]), 
        #                                dim =1
        #                                ).view(ch.shape) 
        #              for ch in out_chs]
        x_net2_in = torch.cat(out_chs, 1)

        # x_net2_in=functional.normalize(
        #                                 x_net2_in.view(-1,x_net2_in.shape[2]*x_net2_in.shape[3]),
        #                                 dim =1
        #                                ).view(x_net2_in.shape)                              
        final_dict = self.net2(x_net2_in)
        conv_out = final_dict[final_dict.keys()[0]]
        conv_out = self.ReLU(conv_out)
        prev_state =  None
        out_chs = []
        final_out ={}
        input_x  = conv_out
        for i in range(self.max_loop_size):
            prev_state = self.clstm(conv_out, prev_state)
            hidden,cell = prev_state
            #input_x =hidden
            out_chs.append(self.compress_Conv(hidden))

        # out_chs2 = []
        # prev_state2 =  None
        # for i in range(self.max_loop_size)[::-1]:
        #     prev_state2 = self.clstm2(out_chs[i], prev_state2)
        #     hidden2,cell2 = prev_state2
        #     out_chs2.append(self.compress_Conv(hidden2))

        
        
        #outputs['final']=torch.cat(out_chs,1)
        #return outputs
        final_out['final']=torch.sigmoid(torch.cat(out_chs,1))
        #final_out['final']=hidden
        return final_out
        
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
    def __init__(self, mnet=None, freeze_net1=False, target_label={'unassigned': 1}, label_catin_net2=None, in_ch=1, out_ch=1, first_out_ch=16, \
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
        self.label_catin_net2 =label_catin_net2
        self.target_label = target_label
        self.add_module('first_' + self.net1.name, self.net1)
        self.add_module('second_' + self.net2.name, self.net2)
        if freeze_net1:
            self.freezeWeight(self.net1)

    @property
    def name(self):
        label_ch_list = self.label_catin_net2 \
                   if self.label_catin_net2 \
                   else self.target_label
        cat_in_lable = '_['+'_'.joint(label_ch_list)+']'
        return 'Mdecoder2Unet' \
               + '_in_{}_chs'.format(self.in_ch) \
               + cat_in_lable

    def freezeWeight(self, net):
        for child in net.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x=self.finnal_conv2d(x)
        # outputs ={}
        outputs = self.net1(x)
        if not self.label_catin_net2:
            out_chs = outputs.values()
        else:
            out_chs= [outputs[lb] for lb in self.label_catin_net2]
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
