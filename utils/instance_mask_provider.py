import os, sys
sys.path.append('../')
import pdb
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt
from utils.EMDataset import labelGenerator
from utils.EMDataset import CRIME_Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoaderIter
from utils.transform import VFlip, HFlip, Rot90, random_transform
import matplotlib
import time
from skimage.color import label2rgb
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch_networks.networks import Unet,DUnet,MdecoderUnet,Mdecoder2Unet, \
                                     MdecoderUnet_withDilatConv, Mdecoder2Unet_withDilatConv


class Proc_DataLoaderIter(DataLoaderIter):
	def __init__(self,loader):
		super(Proc_DataLoaderIter,self).__init__(loader)
	def __make_input_data__(self,data,preds):
	 	'''we can concate preds into data as etra channels
           for noew, we just return data
	 	'''
	 	return data

class NN_proc_DataLoaderIter(Proc_DataLoaderIter):
	def __init__(self,loader,nn_model,use_gpu = False):
		super(NN_proc_DataLoaderIter,self).__init__(loader)
		self.nn_model = nn_model.float()
		self.use_gpu = use_gpu
		if use_gpu:
			self.nn_model  = self.nn_model.cuda()
		self.nn_model.eval()
	
	def __next__(self):
		batch=super(NN_proc_DataLoaderIter,self).__next__()

		data,seg_label, targets = batch
		if self.use_gpu:
			data = data.cuda().float()
		preds           = self.nn_model(Variable(data))
		#print('preds shape ={}'.format(preds['distance'].shape) )
		preds           = dict(map(lambda (k,v): (k, v.data.cpu()), preds.iteritems()))
		data            = data.cpu()

		print('preds 2 shape ={}'.format(preds['distance'].shape) )
		#preds           = variable2tensor_fordict(preds)
		pred_seg        = self.__segment__(preds)
		Mask_in,Mask_gt = self.__make_mask__(pred_seg,seg_label)
		input_data      = self.__make_input_data__(data,preds)

		Mask_in         = torch.from_numpy(Mask_in).float()
		Mask_gt         = torch.from_numpy(Mask_gt).float()

		print('input_data shape ={}'.format(input_data.shape) )
		print('Mask_in shape ={}'.format(Mask_in.shape) )
		print('Mask_gt shape ={}'.format(Mask_gt.shape) )
		input_data      = torch.cat([input_data,Mask_in,pred_seg],dim =1)
		return input_data, Mask_gt

	next = __next__  #''' Compatible with Pyton 2.x '''
	
	def __segment__(self,preds):
		distance = preds['distance']
		# print(type(distance))
		# print(distance.shape)
		# print(distance.dim)
		# print(distance[1])
		assert (distance.dim() ==4 and distance.shape[1] ==1)
		seg2D_list = [ watershed_seg2D(torch.squeeze(distance[i]))
		               for i in range(len(distance))]
		seg_batch  =  torch.stack( 
				       			  [  torch.unsqueeze(torch.Tensor(seg2D),0)
				       			     for seg2D in seg2D_list
				       			  ],
				       			  dim =0
			       			     )
		return seg_batch
	
	def __make_mask__(self, pred_seg_batch, target_seg_batch):
		n_samples      = len(pred_seg_batch)
		mid_slice_idx  = pred_seg_batch.shape[1]//2
		unqiue_ids_in_each_seg      = [np.unique(
			                                     pred_seg_batch[i][mid_slice_idx].cpu().numpy()
			                                    ) 
		                                for i in range(n_samples)
		                              ]

		selected_seg_ids            = [select_nonzero_id(unqiue_ids_in_each_seg[i],pred_seg_batch[i][mid_slice_idx].cpu().numpy())
									     for i in range(n_samples)
									   ]

		masked_preds                =  [(pred_seg_batch[i].cpu().numpy() == selected_seg_ids[i]).astype(np.int) 
		                                for i in range(n_samples)]

		masked_target_ids           =  [find_max_coverage_id(masked_preds[i], np.expand_dims(target_seg_batch[i][mid_slice_idx].cpu().numpy(),axis=0))
		                                for i in range(n_samples)]
		
		masked_target               =   [(target_seg_batch[i].cpu().numpy()== masked_target_ids[i]).astype(np.int)
		                                for i in range(n_samples)]
		maksed_target_batch =  np.stack(masked_target,axis = 0)
		maksed_preds_batch  =  np.expand_dims(np.concatenate(masked_preds,axis =0),axis=1)
		return maksed_preds_batch, maksed_target_batch

	 


class GT_proc_DataLoaderIter(Proc_DataLoaderIter):
	def __init__(self,loader):
		super(GT_proc_DataLoaderIter,self).__init__(loader)
	def __next__(self):
		#print ('gt next......')
		batch=super(GT_proc_DataLoaderIter,self).__next__()
		data,seg_label,targets = batch
		Mask_in,Mask_gt = self.__make_mask__(seg_label)
		input_data      = self.__make_input_data__(data,targets)
		Mask_in         = torch.from_numpy(Mask_in).float()
		Mask_gt         = torch.from_numpy(Mask_gt).float()

		#print ('input shape ={}'.format(input_data.shape))
		#print('Mask_in shape ={}'.format(Mask_in.shape))
		input_data      = torch.cat([input_data,Mask_in],dim =1)
		return input_data, Mask_gt
	
	next = __next__ #''' Compatible with Pyton 2.x '''
	

	def __make_mask__(self,target_seg_batch):

		n_samples = len(target_seg_batch)
		mid_slice_idx  = target_seg_batch.shape[1]//2
		unqiue_ids_in_each_seg      = [np.unique(
												target_seg_batch[i][mid_slice_idx].cpu().numpy()
												)
										 for i in range(n_samples)
		                              ]
		
		selected_seg_ids              = [select_nonzero_id(unqiue_ids_in_each_seg[i],target_seg_batch[i][mid_slice_idx].cpu().numpy())
									     for i in range(n_samples)
									   ]

		
		
		masked_target               =   [ (target_seg_batch[i].cpu().numpy()== selected_seg_ids[i]).astype(np.int) 
		                                for i in range(n_samples)]

		
		masked_preds                = [ np.expand_dims(masked_target[i][mid_slice_idx],0)
		                                for i in range(n_samples)]
		
		maksed_target_batch =  np.stack(masked_target, axis = 0)
		maksed_preds_batch  =  np.stack(masked_preds,axis =0)
		return maksed_preds_batch, maksed_target_batch


class instance_mask_Dataloader(DataLoader):
	def __init__(self,**kwargs):
		super(instance_mask_Dataloader,self).__init__(**kwargs)

	#def __iter__(self):
		'''subclass should return different Dataloaderiter'''
	#	raise NotImplementedError(" __iter__ function must be implemented in subclass !")

	def gen_train_input(data,Preds,Mask_in):
		input_data = torch.cat([data,Preds['distance'], Mask_in],1)
		return input_data



class instance_mask_NNproc_DataLoader(instance_mask_Dataloader):
	def __init__(self,nn_model,use_gpu=False,**kwargs):
		super(instance_mask_NNproc_DataLoader,self).__init__(**kwargs)
		self.nn_model = nn_model
		self.nn_model_use_gpu = use_gpu
	
	def __iter__(self):
		return NN_proc_DataLoaderIter(self,self.nn_model,self.nn_model_use_gpu)



class instance_mask_GTproc_DataLoader(instance_mask_Dataloader):
	def __init__(self,**kwargs):
		super(instance_mask_GTproc_DataLoader,self).__init__(**kwargs)
		#self.CRIME_Dataset_3D = CRIME_Dataset_3D_labels(kwargs)
		#self.data_loader = DataLoader
	def __iter__(self):
		return GT_proc_DataLoaderIter(self)

class CRIME_Dataset_3D_labels(CRIME_Dataset):
	def __init__(self,
                 out_patch_size       =   (224,224,3), 
                 sub_dataset          =   'Set_A',
                 subtract_mean        =   True,
                 phase                =   'train',
                 transform            =   None,
                 data_config          =   '../conf/cremi_datasets.toml'):
	    super(CRIME_Dataset_3D_labels,self).__init__(sub_dataset=sub_dataset, 
                                         out_patch_size =out_patch_size,
                                         subtract_mean =subtract_mean,
                                         phase = phase,
                                         transform =transform,
                                         data_config = data_config)

	def __getitem__(self, index):
		im_data,lb_data= self.random_choice_dataset(self.im_lb_pair)
		data,seg_label = self.get_random_patch(index,im_data,lb_data)
		'''Convert seg_label to 2D by obtaining only intermedia slice
		while the input data have multiple slice as multi-channel input
		the network only output the prediction of sigle slice in the center of Z dim'''
		tc_data        = torch.from_numpy(data).float()
		tc_label_dict  = self.gen_label_per_slice(seg_label)
		return tc_data, seg_label, tc_label_dict 

	def gen_label_per_slice(self,seg):
		if seg.ndim == 3:
			z_dim = seg.shape[0]
			assert ((z_dim % 2) == 1) #  need ensure that # slices is odd number
			slice_seg_list = [seg[i,:,:] for i in range(z_dim)]
			slice_tgDict_list  = self.label_generator(*slice_seg_list)
			lb_dict= {}
			for k in slice_tgDict_list[0].keys():
				lb_dict[k] = torch.cat([slice_tgDict_list[i][k] for i in range(len(slice_seg_list))],dim=0)
			return lb_dict


def watershed_seg2D(distance):
	from scipy import ndimage
	from skimage.feature import peak_local_max
	from skimage.segmentation import watershed
	from skimage.color import label2rgb
	from skimage.morphology import disk,skeletonize
	import skimage
	from skimage.filters import gaussian
	if isinstance(distance,Variable):
		distance = distance.data
	distance = distance.cpu().numpy()
	distance = np.squeeze(distance)
	markers = distance > 3.5
	markers = skimage.morphology.label(markers)
	seg_labels  = watershed(-distance, markers)
	return seg_labels

def tensor_unique(t_tensor):
	t = np.unique(t_tensor.cpu().numpy())
	return torch.from_numpy(t)

def random_select(d_list):
    d_id=np.random.choice(d_list)
    return d_id

def find_max_coverage_id(mask, seg):
	#print('mask shape = {}'.format(mask.shape))
	#print('seg shape ={}'.format(seg.shape))
	bool_mask = mask.astype(np.bool)
	converted_ids=seg[bool_mask]
	unqiue_ids,count = np.unique(converted_ids,return_counts = True)
	#print(unique_ids)
	idex = np.argmax(count)
	return unqiue_ids[idex]

def select_nonzero_id(unique_ids,seg,threshed = 36):
	islarger = False
	while not islarger:
		sid = np.random.choice(unique_ids)
		if sid >0:
		   islarger = np.sum((seg== sid).astype(np.int)) > threshed
	return sid


def variable2tensor_fordict(preds_dict):
    for key,value in preds_dict.iteritems():
        label_dict[key] =value.data
    return label_dict


def test(masker):
	def data_Transform(op_list):
		cur_list = []
		ops = {'vflip':VFlip(),'hflip':HFlip(),'rot90':Rot90()}
		for op_str in op_list:
			cur_list.append(ops[op_str])
		return random_transform(* cur_list)

	def view_output(input_data,mask_target):
		fig = plt.figure()
		y = 4 if len(input_data[0])>3 else 3
		for t  in range(3):
			a = fig.add_subplot(3, y, t*2+1)
			imgplot = plt.imshow(input_data[0,t])
			a.set_title('im')
			a = fig.add_subplot(3, y, t*2+2)
			imgplot = plt.imshow(mask_target[0,t])
			a.set_title('mask')

			a = fig.add_subplot(3, y, t*2+3)
			imgplot = plt.imshow(input_data[0,3])
			a.set_title('mask_in')


			#label2rgb(labels), interpolation='nearest'
			if len(input_data[0])>3:
				a = fig.add_subplot(3, y, t*2+4)
				imgplot = plt.imshow(label2rgb(input_data[0,4].numpy()))
				a.set_title('seg')
		plt.show()
		



	transform = data_Transform(['vflip','hflip','rot90'])

	dataset = CRIME_Dataset_3D_labels(out_patch_size       =   (480,480,3), 
						              sub_dataset          =   'All',
						              subtract_mean        =   True,
						              phase                =   'train',
						              transform            =   transform,
						              data_config          =   '../conf/cremi_datasets_with_tflabels.toml')

	if masker == 'GT_Mask':
		dataLoader = instance_mask_GTproc_DataLoader(dataset    = dataset,
	                                				batch_size  = 10,
	                                				shuffle     = True,
	                                				num_workers = 1)
	elif masker  =='NN_Mask':
		data_out_labels = dataset.output_labels()
		nn_model = Mdecoder2Unet_withDilatConv(target_label = data_out_labels, in_ch = 3)
		nn_model = nn_model.float()
		pre_trained_weights = \
		'../model/Mdecoder2Unet_withDilatConv_in_3_chs_Dataset-CRIME-All_affinity-sizemap-centermap-distance-gradient_VFlip-HFlip-Rot90_freeze_net1=True_iter_32499.model'
		nn_model.load_state_dict(torch.load(pre_trained_weights))
		data_loader = instance_mask_NNproc_DataLoader(nn_model   = nn_model,
		                                                 	use_gpu     = True,
														 	dataset     = dataset,
														 	batch_size  = 5,
														 	shuffle     = True,
														 	num_workers = 1
														 )
	start_time  = time.time()


	for i,(input_data,mask_target) in enumerate(data_loader, 0):
		#plt.imshow(input_data[0,2])''
		end_time = time.time() - start_time
		
		print('time  = {:2} s'.format(end_time))
		print('size of obj = {}'.format( torch.sum(mask_target)))
		
		view_output(input_data, mask_target)
		start_time = time.time()
		
		if i > 20:
			break

if __name__ == '__main__':
  test(masker = 'NN_Mask')


