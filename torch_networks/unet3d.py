import os, sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from utils.torch_loss_functions import angularLoss

import torchvision.models as models
import sys
import math
import pdb
import numpy as np

class DownBlock3D(nn.Module):
	""" This is the downblock construction for 3d convs"""
	def __init__(self, in_ch, z_shape, ch_growth_rate, kernel_size = 3):
		super(DownBlock3D, self).__init__()
		self.in_ch = in_ch
		self.z_shape = z_shape
		self.ch_growth_rate = ch_growth_rate
		self.kernel_size = kernel_size
		self.layers = self.build_layer_block()
		self.block = nn.Sequential(*self.layers)
	def forward(self,x):
		outputs = self.block(x)
		outputs = torch.squeeze(outputs,2)
		return outputs

	def build_layer_block(self):
		layers = []
		same_padding = self.kernel_size//2
		num_conv = (self.z_shape - 1)/2
		print('number of convolution layer is: {}').format(num_conv)
		for i in range(num_conv):
			if i == 0:
				out_ch = self.in_ch * self.ch_growth_rate
			else:
				self.in_ch = out_ch
			layers.append(nn.Conv3d(self.in_ch, out_ch, self.kernel_size, padding=(0,same_padding,same_padding)))
			layers.append(nn.BatchNorm3d(out_ch))
			layers.append(nn.ReLU())
		layers.append(nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), return_indices=False, ceil_mode=False))
		return layers 

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
            layers.append(nn.Conv2d(self.in_ch, out_ch, kernel_size=self.kernel_size, padding=same_padding))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False, ceil_mode=False))
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

class conv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3):
        super(conv_bn_relu,self).__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.same_padding = (kernel_size -1)/2
        self.Conv      = nn.Conv2d(self.in_ch,self.out_ch,self.same_padding)
        self.BatchNorm = nn.BatchNorm2d(out_ch)
        self.ReLU      = nn.ReLU(inplace=True)
    def forward(self,x):
        #x1 = self.Conv2d(x)
        #x1 = self.BatchNorm(x)
        #x1 = self.ReLU(x)
        return self.ReLU(self.BatchNorm(self.Conv(x)))

class Unet(nn.Module):
    def __init__(self, in_ch =1, first_out_ch=16, out_ch =1, number_bolck=4,num_conv_in_block=2,ch_change_rate=2,kernel_size = 3):
        super(Unet, self).__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        #self.ch_down_rate =ch_down_rate
        
        #self.conv_2d_1 = nn.Conv2d(in_ch, first_out_ch, kernel_size=kernel_size, padding=kernel_size //2 )

        self.conv_2d_1 = conv_bn_relu(in_ch, first_out_ch, kernel_size = kernel_size)
        
        self.down_block_1 = Downblock(first_out_ch,num_conv_in_block,ch_change_rate,kernel_size)
        
        b2_down_ch = first_out_ch * ch_change_rate
        self.down_block_2 = Downblock(b2_down_ch,num_conv_in_block,ch_change_rate,kernel_size)
        
        b3_down_ch= b2_down_ch * ch_change_rate
        self.down_block_3 = Downblock(b3_down_ch,num_conv_in_block,ch_change_rate,kernel_size)
        
        b4_down_ch = b3_down_ch * ch_change_rate
        self.down_block_4 = Downblock(b4_down_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b5_down_ch = b4_down_ch * ch_change_rate
        self.down_block_5 = Downblock(b5_down_ch,num_conv_in_block,1,kernel_size)
        # outout ch = 256

        b0_up_ch = b5_down_ch * 1

        # input ch = 512, output ch = 256
        self.up_block_0 = Upblock(b0_up_ch+b5_down_ch,num_conv_in_block,2,kernel_size)
        b1_up_ch = (b0_up_ch+b5_down_ch)// 2
        self.up_block_1 = Upblock(b1_up_ch+b4_down_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b2_up_ch = (b1_up_ch+b4_down_ch) // ch_change_rate
        self.up_block_2 = Upblock(b2_up_ch+b3_down_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b3_up_ch = (b2_up_ch + b3_down_ch) // ch_change_rate
        self.up_block_3 = Upblock(b3_up_ch+b2_down_ch,num_conv_in_block,ch_change_rate,kernel_size)
        #self.up_block_3 = Upblock(160,num_conv_in_block,ch_change_rate,kernel_size)

        b4_up_ch = (b3_up_ch+b2_down_ch) // ch_change_rate
        #self.up_block_4 = Upblock(96,num_conv_in_block,ch_change_rate,kernel_size)
        self.up_block_4 = Upblock(b4_up_ch+first_out_ch,num_conv_in_block,ch_change_rate,kernel_size)

        last_up_ch = (b4_up_ch+first_out_ch) // ch_change_rate
        #self.finnal_conv2d = nn.Conv2d(last_up_ch, out_ch, kernel_size=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        self.finnal_conv2d = nn.Conv2d(last_up_ch, 2, kernel_size=3, padding=1)
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


class _Unet_encoder3D(nn.Module):
    def __init__(self, in_ch =1, first_out_ch=16, z_shape = 3, number_bolck=4, \
                 num_conv_in_block=2,ch_change_rate=2,kernel_size = 3):
        super(_Unet_encoder3D, self).__init__()
        self.in_ch  = in_ch
        #self.out_ch = out_ch
        self.z_shape = z_shape
        # First change the channel to 16
        self.conv_3d_1 = nn.Conv3d(in_ch, first_out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        print('in channel is {} and first out channel is {}').format(in_ch, first_out_ch)
        # 3D Convolutions, the output will be (1, 1/2 h, 1/2 w). z dimension is squzzed
        self.enc_3d = DownBlock3D(first_out_ch, self.z_shape, ch_growth_rate=ch_change_rate, kernel_size=kernel_size)
        b0_down_ch = first_out_ch * ch_change_rate

        self.enc_1 = Downblock(b0_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b1_down_ch = b0_down_ch * ch_change_rate
        
        self.enc_2 = Downblock(b1_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b2_down_ch= b1_down_ch * ch_change_rate

        self.enc_3 = Downblock(b2_down_ch, num_conv_in_block, ch_change_rate, kernel_size)
        b3_down_ch = b2_down_ch * ch_change_rate
        
        self.enc_4 = Downblock(b3_down_ch, num_conv_in_block, 1, kernel_size)
        self.b4_down_ch =b3_down_ch * 1

        # self.enc_5 = Downblock(b4_down_ch, num_conv_in_block, 1, kernel_size)
        # self.b5_down_ch =b4_down_ch * 1
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')

    @property
    def last_ch(self):
        return self.b4_down_ch
    def forward(self,x):
        x3d1  = self.conv_3d_1(x) 
        print('x3d1 shape is {}').format(x3d1.data.shape)
        x0 = x3d1[:,:,(self.z_shape-1)/2,:,:]
        x1 = self.enc_3d(x3d1)
        print('x1 shape is {}').format(x1.data.shape)
        d_1 = self.enc_1(x1)
        d_2 = self.enc_2(d_1)
        d_3 = self.enc_3(d_2)
        d_4 = self.enc_4(d_3)
        # d_5 = self.enc_5(d_4)
        enc3_out = torch.cat((self.upsample(d_4), d_3), 1)
        return [enc3_out,d_2, d_1, x1, x0]


class _Unet_decoder(nn.Module):
    def __init__(self,bottom_input_ch, out_ch =2, num_conv_in_block =2 , ch_change_rate =2 ,kernel_size =3):
        super(_Unet_decoder,self).__init__()
        b1_in_up_ch = bottom_input_ch + bottom_input_ch / 1 
        self.dec_1 = Upblock(b1_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b2_in_up_ch = b1_in_up_ch // ch_change_rate + bottom_input_ch // (2**1)
        self.dec_2 = Upblock(b2_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b3_in_up_ch = b2_in_up_ch // ch_change_rate + bottom_input_ch // (2**2)
        self.dec_3 = Upblock(b3_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b4_in_up_ch = b3_in_up_ch // ch_change_rate + bottom_input_ch // (2**3)
        self.dec_4 = Upblock(b4_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b5_in_up_ch = b4_in_up_ch // ch_change_rate + bottom_input_ch // (2**4)
        self.dec_5 = Upblock(b5_in_up_ch,num_conv_in_block,ch_change_rate,kernel_size)

        last_up_ch = b5_in_up_ch // ch_change_rate
        self.finnal_conv2d = nn.Conv2d(last_up_ch, out_ch, kernel_size=3, padding=(kernel_size-1)/2)

        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
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

         out = self.finnal_conv2d(u_5)
         return out


    
class MdecoderUnet3D(nn.Module):
    def __init__(self, in_ch =1, first_out_ch=16, z_shape = 3, target_label = {'nameless':1}, \
                number_bolck=4, num_conv_in_block=2, ch_change_rate=2,kernel_size = 3):
        super(MdecoderUnet3D,self).__init__()
        self.encoder = _Unet_encoder3D(in_ch, first_out_ch, z_shape, number_bolck, num_conv_in_block, ch_change_rate, kernel_size)
        #self.add_module('encoder',self.encoder)
        
        self.decoders = {}
        for name,out_ch in target_label.iteritems():
            self.decoders[name]=_Unet_decoder(bottom_input_ch=self.encoder.last_ch, out_ch=out_ch,num_conv_in_block = num_conv_in_block)
            self.add_module('decoder_'+ name, self.decoders[name])

    def forward(self,x):
        encoder_outputs= self.encoder(x)
        outputs = {}
        for name, decoder in self.decoders.iteritems():
            outputs[name]=decoder(encoder_outputs)
        return outputs
    @property
    def name(self):
        return 'MdecoderUnet3D'


class DUnet3D(nn.Module):
    def __init__(self, grad_unet, freeze_net1 =True, in_ch =1, first_out_ch=16, out_ch =1, number_bolck=4,num_conv_in_block=2,ch_change_rate=2,z_shape=3,kernel_size = 3):
        super(DUnet3D, self).__init__()
        self.net1 = grad_unet
        self.net2 = Unet()
        self.first_conv_in_net2 = nn.Conv2d(3,16,kernel_size=kernel_size,padding =kernel_size // 2)
        self.final_conv_in_net2 = nn.Conv2d(48,out_ch,kernel_size=kernel_size,padding =kernel_size // 2) 
        self.net2.conv_2d_1 = self.first_conv_in_net2
        self.net2.finnal_conv2d = self.final_conv_in_net2
        self.out_ch = out_ch
        self.z_shape=z_shape
        if freeze_net1:
            self.freezeWeight(self.net1)
    @property
    def name(self):
        return 'DUnet3D_outch_{}'.format(self.out_ch)
    def freezeWeight(self,net):
        for child in net.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self,x):
        #x=self.finnal_conv2d(x)
        outputs ={}
        outputs['gradient']   = self.net1(x)['gradient']
        x_net2_in             = torch.cat((outputs['gradient'],x[:,:,(self.z_shape-1)/2,:,:]),1)
        outputs['distance']   = self.net2(x_net2_in)
        return outputs
        #return gradient_out,distance_out



if __name__ == '__main__':
    # net = DownBlock3D(in_ch=16, z_shape=3, ch_growth_rate=2)
    # net = nn.Conv3d(16, 32, kernel_size = 3)
    net1 = MdecoderUnet3D(z_shape = 3, target_label = {'gradient':2})
    net = DUnet3D(net1,freeze_net1=False)
    inputs = Variable(torch.randn(3,1,3,320,320))
    outputs = net(inputs)
    # outputs = torch.squeeze(outputs)
    # print(outputs.shape)
    
