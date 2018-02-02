import os, sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import math
import pdb
import numpy as np


class hybrid_2d3d_unet(nn.Module):
	def __init__(self,in_ch=1, out_ch=1, first_out_ch =16, act_fn =None, target_label={'distance':1}, BatchNorm_final =None):
		super(hybrid_2d3d_unet, self).__init__()
		encoder_blocks  =4
		if act_fn  is  None:
			act_fn = nn.LeakyReLU(0.2, inplace=True)
		self.encoder = __encoder__(in_ch,
									first_out_ch,
									down_block,
									act_fn,
									num_blocks =encoder_blocks)
		bottom_in_ch = first_out_ch * 2**encoder_blocks
		self.decoder = __decoder__(bottom_in_ch, 
			                             out_ch,
			                             up_block,
			                             act_fn,
			                             num_blocks =encoder_blocks)
		self.add_module('encoder',self.encoder)
		self.add_module('decoder',self.decoder)
	
	def forward(self, x):
		out_list=self.encoder(x)
		num_chs = [out.size() for out in out_list]
		print(num_chs)
		x=self.decoder(out_list)
		return {'distance':x}


class __encoder__(nn.Module):
	def __init__(self, in_ch, first_out_ch, down_block_fn, act_fn, num_blocks =1, res_reduce_factor =2):
		super(__encoder__, self).__init__()
		self.resolution_reduce_conv = nn.Conv3d(in_ch,first_out_ch, 
			                               kernel_size = (1,4,4),
			                               padding = (0,1,1),
			                               stride=(1, res_reduce_factor, res_reduce_factor)
			                               )
		self.encode_fn_list = nn.ModuleList()
		in_ch = first_out_ch
		out_ch = in_ch *2
		block_fn = res_conv_hybrid2D3D_block
		for i in range (num_blocks):
			self.encode_fn_list.append(down_block_fn(in_ch, out_ch, block_fn, act_fn))
			in_ch  = out_ch
			out_ch *= 2
	
	def forward(self, x):
		encode_out_list = []
		x = self.resolution_reduce_conv(x)
		encode_out_list.append(x)
		for idx, block_encoder in enumerate(self.encode_fn_list):
			#input_x = x if idx == 0 else encode_out_list[idx-1]
			input_x = encode_out_list[idx]
			encode_out_list.append(block_encoder(input_x))
		return encode_out_list


class __decoder__(nn.Module):
	def __init__(self, bottom_in_ch, last_out_ch, up_block_fn, act_fn, num_blocks =1, res_upsample_factor =2):
		super(__decoder__, self).__init__()
		# self.space__conv = nn.Conv3d(in_ch,first_out_ch, 
		# 	                               kernel_size = (1,4,4),
		# 	                               padding = (0,2,2),
		# 	                               stride=(1, res_reduce_factor, res_reduce_factor)
		# 	                               )
		self.decode_fn_list = nn.ModuleList()
		self.upsample = upsample2D()
		self.last_upsample_conv= nn.Sequential(
			                       self.upsample,
			                       conv_block_2D(bottom_in_ch / (2** (num_blocks)), 
			                                     last_out_ch,
			                                     act_fn)
			                          )
		in_ch = bottom_in_ch + bottom_in_ch /2
		out_ch = bottom_in_ch /2
		block_fn = res_conv_hybrid2D3D_block
		for i in range (num_blocks):
			self.decode_fn_list.append(up_block_fn(in_ch, out_ch, block_fn, act_fn))
			in_ch  = out_ch + out_ch/2
			out_ch /=2
	
	def forward(self, encoder_outputs):
		#encode_out_list = []
		encoder_outputs = encoder_outputs[::-1]
		for idx, block_decoder in enumerate(self.decode_fn_list):
			if idx  ==0:
			 	x = block_decoder(torch.cat([self.upsample(encoder_outputs[0]),encoder_outputs[1]],1))
			else:
			# 	x = torch.cat((self.upsample(x),encoder_outputs[idx]),1)
			# 	x = block_decoder(x)
			    x = torch.cat((self.upsample(x),encoder_outputs[idx+1]),1)
			    x = block_decoder(x)
		x = self.last_upsample_conv(x)

		return x



class down_block(nn.Module):
	def __init__(self, in_ch, out_ch, block_fn, act_fn, repeat =1):
		super(down_block, self).__init__()
		#self.block = block_fn
		block_list =[]
		for i in range(repeat):
			bin_ch = in_ch  if i ==0 else out_ch
			block_list.append(block_fn(bin_ch, out_ch, act_fn))
		self.block = nn.Sequential(*block_list)
		self.pool = maxpool_2D()
	
	def forward(self, x):
		x = self.block(x)
		x = self.pool(x)
		return x


class up_block(nn.Module):
	def __init__(self, in_ch, out_ch, block_fn, act_fn, repeat =1):
		super(up_block, self).__init__()
		#self.block = block_fn
		#self.upsample = upsample2D()
		block_list =[]
		for i in range(repeat):
			bin_ch = in_ch  if i ==0 else out_ch
			block_list.append(block_fn(bin_ch, out_ch, act_fn))
		self.block = nn.Sequential(*block_list)
		#self.pool = maxpool_2D()
	
	def forward(self, x):
		#x = self.upsample(x)
		x = self.block(x)
		return x




class res_conv_hybrid2D3D_block(nn.Module):
	def __init__(self, in_ch, out_ch, act_fn):
		super(res_conv_hybrid2D3D_block, self).__init__()
		self.in_ch, self.out_ch, self.act_fn  = in_ch, out_ch, act_fn
		self.res_branch = conv_hybrid2D3D_block(in_ch,out_ch,act_fn)
		self.skip_branch = nn.Conv3d(in_ch,out_ch, kernel_size =1, padding=0)
	def forward(self,x):
		if self.in_ch == self.out_ch:
			out = self.res_branch(x) + x
		else:
			out =self.res_branch(x)
			out += self.skip_branch(x)
		out = self.act_fn(out)
		return out

def conv_hybrid2D3D_block(in_ch,out_ch,act_fn):
	branch = nn.Sequential(
		conv_block_2D(in_ch, out_ch, act_fn),
		conv_block_3D(out_ch, out_ch, act_fn),
		nn.Conv3d(out_ch,out_ch,kernel_size =(1,3,3), padding=(0,1,1),stride =1),
		nn.BatchNorm3d(out_ch),
		)
	return branch

def conv_2D3D_hybrid(in_ch,out_ch,act_fn):
	return nn.Sequential(
		conv_block_2D(in_ch, out_ch, act_fn),
		conv_block_3D(out_ch, out_ch, act_fn),
		nn.Conv3d(out_ch,out_ch,kernel_size =(1,3,3), padding=(0,1,1),stride =1),
		nn.BatchNorm3d(out_ch),
		)

def conv_block_3D(in_ch,out_ch,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_ch,out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_ch),
        act_fn
    )
    return model


def conv_block_2D(in_ch, out_ch, act_fn):
	kernel_sizes=(1,3,3)
	paddings =(0,1,1)
	return nn.Sequential(
		nn.Conv3d(in_ch,out_ch,kernel_size =kernel_sizes, padding=paddings,stride =1),
		nn.BatchNorm3d(out_ch)
		)

def maxpool_3D():
	return nn.nn.MaxPool3d(kernel_size=2,stride=2,padding=0)

def maxpool_2D():
	kernel_sizes=(1,2,2)
	strides =(1,2,2)
	return nn.MaxPool3d(kernel_size=kernel_sizes,stride=strides,padding=0)

def upsample3D():
	return nn.Upsample(scale_factor=2, mode='trilinear')

def upsample2D():
	return nn.Upsample(scale_factor=(1,2,2), mode='trilinear')

def test():
	use_gpu  =True
	xy_size = 320
	z_size  = 25
	act_fn = nn.LeakyReLU(0.2, inplace=True)
	x = Variable(torch.Tensor(1,1,z_size,xy_size,xy_size))
	#model = conv_block_3D(1, 5, act_fn)
	#model = res_conv_hybrid2D3D_block(1,5,act_fn)
	#model = down_block(1,5,res_conv_hybrid2D3D_block,act_fn, repeat =3)

	#model = __encoder__(1,16, down_block, act_fn, num_blocks =4, res_reduce_factor =2)
	model =hybrid_2d3d_unet(1,1,act_fn = act_fn)
	
	if use_gpu and torch.cuda.is_available():
		model = model.cuda().float()
		x=x.cuda().float()
	
	outlist=model(x)
	for out in outlist:
		print(out.size())
	#print(out[0].size())

if __name__ == '__main__':
	test()
