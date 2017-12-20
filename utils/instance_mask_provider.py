import os, sys
sys.path.append('../')
import pdb
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from utils.EMDataset import labelGenerator
class instance_mask_provider:
	def __init__(data_loader,nn_model,segmentor,mask_maker):
		self.data_loaders   = dataloaders
		self.nn_model       = nn_model
		self.segmentor      = segmentor
		self.mask_maker     = mask_maker
	def generate(phase= 'train'):
		#for i, (data,targets) in enumerate(self.data_loader[phase], 0):
		#	Preds            = self.nn_model(data)
		#	segments         = self.segmentor(Preds)
		#	Mask_in, Mask_gt = self.mask_maker(segments)

		Iter = self.dataloaders[phase].__iter__()
		(data,targets)   = Iter.next()
		Preds            = self.nn_model(data)
		segments         = self.segmentor(Preds)
		Mask_in, Mask_gt = self.mask_maker(segments)
		input_data       = self.gen_train_input(data,Preds,Mask_in)
		return input_data, Mask_gt
	
	def gen_train_input(data,Preds,Mask_in):
		input_data = torch.cat([data,Preds['distance'], Mask_in],1)
		return input_data

class (object):
	def __call__(self,*input):
		raise NotImplementedError("Must be implemented in subclass !")
class label_creator_from

