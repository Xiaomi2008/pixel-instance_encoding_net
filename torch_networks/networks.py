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
class Downblock(nn.Module):
    def __init__(self,in_ch,num_conv,ch_growth_rate,kernel_size = 3):
        super(Downblock, self).__init__()
        assert(num_conv>0)
        self.in_ch = in_ch
        self.num_conv =num_conv
        self.ch_growth_rate =ch_growth_rate
        self.kernel_size =kernel_size
        self.layers=self.build_layer_block()
        self.block = nn.Sequential(*self.layers)
    def forward(self,x):
        return self.block(x)

    def build_layer_block(self):
        layers = []
        same_padding = self.kernel_size // 2
        
        for i in range(self.num_conv):
            if i == 0:
                out_ch = self.in_ch * self.ch_growth_rate
            else:
                self.in_ch = out_ch
            # conv = nn.Conv2d(self.in_ch, out_ch,self.kernel_size=kernel_size, padding=same_padding)
            # print self.in_ch, out_ch
            # conv = nn.Conv2d(self.in_ch, out_ch)
            layers.append(nn.Conv2d(self.in_ch, out_ch, kernel_size=self.kernel_size, padding=same_padding))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))
        return layers


class Upblock(nn.Module):
    def __init__(self, in_ch, num_conv,ch_down_rate,kernel_size = 3):
        super(Upblock, self).__init__()
        assert(num_conv>0)
        self.in_ch = in_ch
        self.num_conv =num_conv
        self.ch_down_rate =ch_down_rate
        self.kernel_size =kernel_size
        self.layers=self.build_layer_block()
        self.block = nn.Sequential(*self.layers)

    def forward(self,x):
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
            print(self.in_ch,self.kernel_size,same_padding)
            layers.append(nn.Conv2d(self.in_ch, out_ch, kernel_size=self.kernel_size, padding=same_padding))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
        return layers

class Unet(nn.Module):
    def __init__(self, in_ch =1, first_out_ch=16, out_ch =1, number_bolck=4,num_conv_in_block=2,ch_change_rate=2,kernel_size = 3):
        super(Unet, self).__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        #self.ch_down_rate =ch_down_rate
        
        self.conv_2d_1 = nn.Conv2d(in_ch, first_out_ch, kernel_size=kernel_size, padding=kernel_size //2 )
        
        self.down_block_1 = Downblock(first_out_ch,num_conv_in_block,ch_change_rate,kernel_size)
        
        b2_down_ch = first_out_ch * ch_change_rate
        self.down_block_2 = Downblock(b2_down_ch,num_conv_in_block,ch_change_rate,kernel_size)
        
        b3_down_ch= b2_down_ch * ch_change_rate
        self.down_block_3 = Downblock(b3_down_ch,num_conv_in_block,ch_change_rate,kernel_size)
        
        b4_down_ch = b3_down_ch * ch_change_rate
        self.down_block_4 = Downblock(b4_down_ch,num_conv_in_block,ch_change_rate,kernel_size)
        

        b1_up_ch = b4_down_ch * ch_change_rate
        self.up_block_1 = Upblock(b1_up_ch+b4_down_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b2_up_ch = b1_up_ch // ch_change_rate
        self.up_block_2 = Upblock(b2_up_ch+b4_down_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b3_up_ch = b2_up_ch // ch_change_rate
        self.up_block_3 = Upblock(160,num_conv_in_block,ch_change_rate,kernel_size)

        b4_up_ch = b3_up_ch // ch_change_rate
        self.up_block_4 = Upblock(96,num_conv_in_block,ch_change_rate,kernel_size)

        last_up_ch = b4_up_ch // ch_change_rate
        #self.finnal_conv2d = nn.Conv2d(last_up_ch, out_ch, kernel_size=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        self.finnal_conv2d = nn.Conv2d(48, 2, kernel_size=3, padding=1)
    @property
    def name(self):
        return 'Unet'
    def forward(self,x):
        #x=self.finnal_conv2d(x)
        x1  = self.conv_2d_1(x)
        d_1 = self.down_block_1(x1)
        d_2 = self.down_block_2(d_1)
        d_3 = self.down_block_3(d_2)
        d_4 = self.down_block_4(d_3)
        c_1 = torch.cat((self.upsample(d_4), d_3), 1)
        u_1 = self.up_block_1(c_1)

        c_2 = torch.cat((self.upsample(u_1), d_2), 1)
        u_2 = self.up_block_2(c_2)

        c_3 = torch.cat((self.upsample(u_2), d_1), 1)
        u_3 = self.up_block_3(c_3)

        c_4 = torch.cat((self.upsample(u_3), x1), 1)
        u_4 = self.up_block_4(c_4)

        out = self.finnal_conv2d(u_4)
        return out


class DUnet(nn.Module):
    def __init__(self, grad_unet, freeze_net1 =True, in_ch =1, first_out_ch=16, out_ch =1, number_bolck=4,num_conv_in_block=2,ch_change_rate=2,kernel_size = 3):
        super(DUnet, self).__init__()
        self.net1 = grad_unet
        self.net2 = Unet()
        self.first_conv_in_net2 = nn.Conv2d(3,16,kernel_size=kernel_size,padding =kernel_size // 2)
        self.final_conv_in_net2 = nn.Conv2d(48,1,kernel_size=kernel_size,padding =kernel_size // 2) 
        self.net2.conv_2d_1 = self.first_conv_in_net2
        self.net2.finnal_conv2d = self.final_conv_in_net2
        if freeze_net1:
            self.freezeWeight(self.net1)
    @property
    def name(self):
        return 'DUnet'
    def freezeWeight(self,net):
        for child in net.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self,x):
        #x=self.finnal_conv2d(x)
        gradient_out = self.net1(x)
        x_net2_in  = torch.cat((gradient_out,x),1)
        distance_out        = self.net2(x_net2_in)
        return gradient_out,distance_out




def test_angularLoss():
    A = torch.randn(16,2,224,224).double()
    B = torch.randn(16,2,224,224).double()
    B = A
    W = torch.randn(1,1,3,3).double()
    W = torch.abs(l2_norm(W))
    print(angularLoss(A,B,W))


if __name__ == '__main__':
    test_angularLoss()
    # net = Unet()
    # print(net)

    # test_x = Variable(torch.FloatTensor(1, 1, 1024, 1024))
    # out_x = net(test_x)

    # print(out_x.size())
