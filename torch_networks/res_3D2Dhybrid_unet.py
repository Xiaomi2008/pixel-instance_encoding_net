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


class res_conv_hybrid2D3D_3section_block(nn.Module):
	def __init__(self, in_ch, out_ch, act_fn):
		super(res_conv_hybrid2D3D_3section_block, self).__init__()
		self.in_ch, self.out_ch, self.act_fn  = in_ch, out_ch, act_fn
		self.res_branch_1 = conv_hybrid2D3D_block(in_ch,out_ch,act_fn,plane='xy')
		self.res_branch_2 = conv_hybrid2D3D_block(in_ch,out_ch,act_fn,plane='xz')
		self.res_branch_3 = conv_hybrid2D3D_block(in_ch,out_ch,act_fn,plane='yz')
		self.skip_branch = nn.Conv3d(in_ch,out_ch, kernel_size =1, padding=0)
		#self.add_module('res1',self.res_branch_1)
	
	def forward(self,x):
		sum_branch = self.res_branch_1(x) +self.res_branch_2(x)+self.res_branch_3(x)
		if self.in_ch == self.out_ch:
			out = sum_branch+x
		else:
			out =sum_branch
			out += self.skip_branch(x)
		out = self.act_fn(out)
		return out



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


class res_conv_hybrid2D3D_3section_block(nn.Module):
	def __init__(self, in_ch, out_ch, act_fn):
		super(res_conv_hybrid2D3D_3section_block, self).__init__()
		self.in_ch, self.out_ch, self.act_fn  = in_ch, out_ch, act_fn
		self.res_branch_1 = conv_hybrid2D3D_block(in_ch,out_ch,act_fn,plane='xy')
		self.res_branch_2 = conv_hybrid2D3D_block(in_ch,out_ch,act_fn,plane='xz')
		self.res_branch_3 = conv_hybrid2D3D_block(in_ch,out_ch,act_fn,plane='yz')
		self.skip_branch = nn.Conv3d(in_ch,out_ch, kernel_size =1, padding=0)
		#self.add_module('res1',self.res_branch_1)
	
	def forward(self,x):
		sum_branch = self.res_branch_1(x) +self.res_branch_2(x)+self.res_branch_3(x)
		if self.in_ch == self.out_ch:
			out = sum_branch+x
		else:
			out =sum_branch
			out += self.skip_branch(x)
		out = self.act_fn(out)
		return out

def conv_hybrid2D3D_block(in_ch,out_ch,act_fn, plane = 'xy'):
	branch = nn.Sequential(
		conv_block_2D(in_ch, out_ch, act_fn, plane=plane),
		conv_block_3D(out_ch, out_ch, act_fn),
		# ''' recent just changed here, to load the old modem just comment this line and enable next line '''
		#conv_block_2D(out_ch, out_ch, act_fn=None,plane=plane),
		nn.Conv3d(out_ch,out_ch,kernel_size =(1,3,3), padding=(0,1,1),stride =1),
		nn.BatchNorm3d(out_ch),
		)
	return branch

# def conv_3section_hybrid2D3D_block(in_ch,out_ch,act_fn):
# 	branch = nn.Sequential(
# 		conv_block_2D(in_ch, out_ch, act_fn),
# 		conv_block_3D(out_ch, out_ch, act_fn),
# 		conv_block_2D(out_ch, out_ch, act_fn=None),
# 		# nn.Conv3d(out_ch,out_ch,kernel_size =(1,3,3), padding=(0,1,1),stride =1),
# 		# nn.BatchNorm3d(out_ch),
# 		)
# 	return branch

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


def conv_block_2D(in_ch, out_ch, act_fn=None, plane='xy'):
	if plane =='xy':
		kernel_sizes=(1,3,3)
		paddings =(0,1,1)
	elif plane =='yz':
		kernel_sizes=(3,3,1)
		paddings =(1,1,0)
	elif plane == 'xz':
		kernel_sizes=(3,1,3)
		paddings =(1,0,1)
	if act_fn:
		return nn.Sequential(
			nn.Conv3d(in_ch,out_ch,kernel_size =kernel_sizes, padding=paddings,stride =1),
			nn.BatchNorm3d(out_ch),
			act_fn
		)
	else:
		return nn.Sequential(
			nn.Conv3d(in_ch,out_ch,kernel_size =kernel_sizes, padding=paddings,stride =1),
			nn.BatchNorm3d(out_ch)
		)


def deconv_block_2D(in_ch,out_ch,act_fn):
	kernel_sizes=(1,4,4)
	paddings =(0,1,1)
	strides  =(1,2,2)
	return nn.Sequential(
		nn.ConvTranspose3d(in_ch,out_ch,kernel_size =kernel_sizes, padding=paddings,stride =strides),
		nn.BatchNorm3d(out_ch),
		act_fn
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
# def deconv_up2D():
# 	return nn.ConvTranspose2d()



class hybrid_2d3d_unet(nn.Module):
	def __init__(self,in_ch=1, out_ch=1, first_out_ch =16, act_fn =None, target_label={'affinity':1}, 
		         BatchNorm_final =None, block_fn=res_conv_hybrid2D3D_block):
		super(hybrid_2d3d_unet, self).__init__()
		encoder_blocks  =4
		if act_fn  is  None:
			act_fn = nn.LeakyReLU(0.2, inplace=True)
		print('activate function is {}'.format(type(act_fn)))
			#act_fn = nn.ReLU6(inplace=True)
			#act_fn = lambda x: x*torch.Sigmoid(x)
			#act_fn = __self_gated_activation__()
		self.encoder = __encoder__(in_ch,
									first_out_ch,
									down_block,
									act_fn,
									num_blocks =encoder_blocks,
									block_fn = block_fn)
		bottom_in_ch = first_out_ch * 2**encoder_blocks
		self.decoder = __decoder__(bottom_in_ch, 
			                             out_ch,
			                             up_block,
			                             act_fn,
			                             num_blocks =encoder_blocks,
			                             block_fn=block_fn)
		self.add_module('encoder',self.encoder)
		self.add_module('decoder',self.decoder)
	
	def forward(self, x):
		encoder_out=self.encoder(x)
		#num_chs = [out.size() for out in out_list]
		#print(num_chs)
		x=self.decoder(encoder_out)
		return {'affinity':x}


	def set_multi_gpus(self, device_ids):
		self.encoder = nn.DataParallel(self.encoder, device_ids).cuda()
		self.decoder = nn.DataParallel(self.decoder,device_ids).cuda()

	@property
	def name(self):
		return 'Res_3D_2D_HybribUnet_'+str(type(self.act_fn)).split('.')[-1]

class __encoder__(nn.Module):
	def __init__(self, in_ch, first_out_ch, down_block_fn, act_fn, num_blocks =1, 
		         res_reduce_factor =2,block_fn =res_conv_hybrid2D3D_block):
		super(__encoder__, self).__init__()
		self.resolution_reduce_conv = nn.Conv3d(in_ch,first_out_ch, 
			                               kernel_size = (1,4,4),
			                               padding = (0,1,1),
			                               stride=(1, res_reduce_factor, res_reduce_factor)
			                               )
		self.encode_fn_list = nn.ModuleList()
		in_ch = first_out_ch
		out_ch = in_ch *2
		#block_fn = res_conv_hybrid2D3D_block
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
	def __init__(self, bottom_in_ch, last_out_ch, up_block_fn, act_fn, num_blocks =1, 
		         res_upsample_factor =2, block_fn = res_conv_hybrid2D3D_block):
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
			                                     act_fn=None)
			                          )
		in_ch = bottom_in_ch + bottom_in_ch /2
		out_ch = bottom_in_ch /2
		#block_fn = res_conv_hybrid2D3D_block
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


class __multihead_decoder__(nn.Module):
	def __init__(self, bottom_in_ch, output_name_chs_dict, up_block_fn, act_fn, num_blocks =1, 
		         res_upsample_factor =2, block_fn=res_conv_hybrid2D3D_block):
		super(__multihead_decoder__, self).__init__()
		self.decode_fn_list = nn.ModuleList()
		deconv_in_ch =bottom_in_ch / (2** (num_blocks))
		self.upsample = upsample2D()

		self.multi_head_fn_list = nn.ModuleList()
		self.out_dict_module ={}
		
		for name,chs in output_name_chs_dict.iteritems():
			# module= nn.Sequential( deconv_block_2D(deconv_in_ch,deconv_in_ch,act_fn),
			# 		            			      conv_block_2D(deconv_in_ch,chs,act_fn=None)
			# 		                            )

			module = nn.Sequential(
				                    conv_block_2D(deconv_in_ch,deconv_in_ch,act_fn=act_fn),
				                    conv_block_2D(deconv_in_ch,chs,act_fn=act_fn),
				                    self.upsample,
				                    conv_block_2D(chs,chs,act_fn=None),
				                   )
			self.multi_head_fn_list.append(module)
			self.out_dict_module[name] = module
		

		in_ch = bottom_in_ch + bottom_in_ch /2
		out_ch = bottom_in_ch /2
		#block_fn = res_conv_hybrid2D3D_block
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
			    x = torch.cat((self.upsample(x), encoder_outputs[idx+1]),1)
			    x = block_decoder(x)
		
		out_dict ={}
		
		for name, module in self.out_dict_module.iteritems():
			out_dict[name] = module(x)

		return out_dict



class hybrid_2d3d_unet_mutlihead(nn.Module):
	def __init__(self,in_ch=1, first_out_ch =16, act_fn =None, target_label={'affinity':1}, 
		         BatchNorm_final =None,block_fn =res_conv_hybrid2D3D_block):
		super(hybrid_2d3d_unet_mutlihead, self).__init__()
		encoder_blocks  =4
		if act_fn  is  None:
			#act_fn = nn.LeakyReLU(0.2, inplace=True)
			#act_fn = nn.ReLU6(inplace=True)
			#act_fn = lambda x: x*torch.Sigmoid(x)
			act_fn = __self_gated_activation__()
		self.encoder = __encoder__(in_ch,
									first_out_ch,
									down_block,
									act_fn,
									num_blocks =encoder_blocks,
									block_fn=block_fn)
		bottom_in_ch = first_out_ch * 2**encoder_blocks
		self.decoder = __multihead_decoder__(bottom_in_ch, 
			                             target_label,
			                             up_block,
			                             act_fn,
			                             num_blocks =encoder_blocks,
			                             block_fn=block_fn)
		self.add_module('encoder',self.encoder)
		self.add_module('decoder',self.decoder)

		print('Res_3D_2D_HybribUnet_Line_81 = {}'.format(target_label))
	
	def forward(self, x):
		encoder_out=self.encoder(x)
		#num_chs = [out.size() for out in out_list]
		#print(num_chs)
		x=self.decoder(encoder_out)
		return x
		#return {'affinity':x}


	def set_multi_gpus(self, device_ids):
		self.encoder = nn.DataParallel(self.encoder, device_ids).cuda()
		self.decoder = nn.DataParallel(self.decoder,device_ids).cuda()

	@property
	def name(self):
		return 'Res_3D_2D_HybribUnet_multiHead_selfGated_act'

class __self_gated_activation__(nn.Module):
	def __init__(self):
		super(__self_gated_activation__, self).__init__()
		#act_fn = nn.Sigmoid()
	def forward(self,x):
		return x* torch.sigmoid(x)


class hybrid_2d3d_unet_mutlihead_with_3section_conv(hybrid_2d3d_unet_mutlihead):
 	def __init__(self,in_ch=1, first_out_ch =16, act_fn =None, target_label={'affinity':1}, 
		         BatchNorm_final =None, block_fn =res_conv_hybrid2D3D_3section_block):
 	    super(hybrid_2d3d_unet_mutlihead_with_3section_conv, self).__init__(
 	    	in_ch=in_ch, first_out_ch=first_out_ch, act_fn=act_fn, target_label=target_label, 
		    BatchNorm_final =BatchNorm_final, block_fn =block_fn)

 	@property
	def name(self):
		return 'Res_3D_2D_HybribUnet_multiHead_sectionCov_selfGated_act'
















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
	#model =hybrid_2d3d_unet(1,1,act_fn = act_fn)
	model =hybrid_2d3d_unet_mutlihead(1,1,target_dict={'affinity':1, 'gradient':2,'distance':1})
	
	if use_gpu and torch.cuda.is_available():
		model = model.cuda().float()
		x=x.cuda().float()
	
	outdict=model(x)
	for k,v in outdict.iteritems():
		print('name , size ={},{}'.format(k,v.size()))
	#print(out[0].size())

if __name__ == '__main__':
	test()
