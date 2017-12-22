import os, sys
sys.path.append('../')
import pdb
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from label_transform.volumes import Volume
from label_transform.volumes import HDF5Volume
from EMDataset import labelGenerator
from EMDataset import exp_Dataset
from mask_utils import default_collate, BatchSampler
from torch.utils.data.sampler import RandomSampler 
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.morphology import disk,skeletonize
import skimage
# from skimage.morphology.skeletonize
# from skimage.filters import gaussian

# class instance_mask_provider:
# 	def __init__(data_loader,nn_model,segmentor,mask_maker):
# 		self.data_loaders   = dataloaders
# 		self.nn_model       = nn_model
# 		self.segmentor      = segmentor
# 		self.mask_maker     = mask_maker
# 	def generate(phase= 'train'):
# 		#for i, (data,targets) in enumerate(self.data_loader[phase], 0):
# 		#	Preds            = self.nn_model(data)
# 		#	segments         = self.segmentor(Preds)
# 		#	Mask_in, Mask_gt = self.mask_maker(segments)

# 		Iter = self.dataloaders[phase].__iter__()
# 		(data,targets)   = Iter.next()
# 		Preds            = self.nn_model(data)
# 		segments         = self.segmentor(Preds)
# 		Mask_in, Mask_gt = self.mask_maker(segments)
# 		input_data       = self.gen_train_input(data,Preds,Mask_in)
# 		return input_data, Mask_gt
	
# 	def gen_train_input(data,Preds,Mask_in):
# 		input_data = torch.cat([data,Preds['distance'], Mask_in],1)
# 		return input_data



class MaskDataset(exp_Dataset):
	def __init__(self,
                 out_patch_size       =   (224,224,1), 
                 sub_dataset          =   'Set_A',
                 subtract_mean        =   True,
                 phase                =   'train',
                 transform            =   None,
                 data_config          =   'conf/cremi_datasets_with_tflabels.toml'):
		self.data_config = data_config
		super(MaskDataset,self).__init__(sub_dataset=sub_dataset, 
                                              out_patch_size =out_patch_size,
                                              subtract_mean =subtract_mean,
                                              phase = phase,
                                              transform =transform)
	def __getitem__(self, index):
		im_data,lb_data= self.random_choice_dataset(self.im_lb_pair)
		data,seg_label = self.get_random_patch(index,im_data,lb_data)
		mask_label = self.mask_gt_maker(seg_label)

		if seg_label.ndim ==3:
			z_dim = seg_label.shape[0]
			assert ((z_dim % 2) == 1)
			m_slice_idx = z_dim // 2
			seg_label_middle = np.expand_dims(seg_label[m_slice_idx,:,:],axis =0) 

		tc_data        = torch.from_numpy(data).float()
		tc_label_dict  = self.label_generator(seg_label_middle)[0]
		tc_label_dict['mask_label'] = torch.from_numpy(mask_label).float()
		tc_label_dict['seg_label'] = torch.from_numpy(seg_label).float()
		return tc_data, tc_label_dict

	def mask_gt_maker(self,seg_label):
		if seg_label.shape[0] > 1:
			middle_slice = seg_label[seg_label.shape[0]//2,:,:]
		else:
			middle_slice = seg_label

		s_id = np.unique(middle_slice).tolist()
		obj_id = np.random.choice(s_id, 1)
		obj_arr = (seg_label == obj_id).astype(int)

		return obj_arr

	def set_phase(self,phase):
		self.phase = phase
		if phase == 'train':
			self.slice_start_z= 0
			self.slice_end_z   = 99
		elif phase == 'valid':
			self.slice_start_z = 100
			self.slice_end_z = 124

		self.z_size = self.slice_end_z - self.slice_start_z +1

	def load_data(self):
		volumes = HDF5Volume.from_toml(self.data_config)
		data_name = {'Set_A':'Sample A with extra transformed labels',
                     'Set_B':'Sample B with extra transformed labels',
                     'Set_C':'Sample C with extra transformed labels'
                    }
      # data_name = {'Set_A':'Sample A',
      #              'Set_B':'Sample B',
      #              'Set_C':'Sample C'
      #             }
		im_lb_pair ={}
		if self.sub_dataset == 'All':
			for k,v in data_name.iteritems():
				V = volumes[data_name[k]]
				im_lb_pair[k] ={'image':V.data_dict['image_dataset'],
								'label':V.data_dict['label_dataset']}
		else:
			k = self.sub_dataset
			V = volumes[data_name[k]]
			im_lb_pair[k] ={'image':V.data_dict['image_dataset'],
							'label':V.data_dict['label_dataset']}

		return im_lb_pair


class instance_mask_provider(object):
	def __init__(self, mask_label_generator, mask_dataset, batch_size):
		self.gen_label = mask_label_generator
		self.batch_size = batch_size
		self.dataset = mask_dataset
	def generate(self):
		sampler = RandomSampler(self.dataset)
		batch_sampler = BatchSampler(sampler, batch_size = self.batch_size , drop_last = False)
		sample_iter = iter(batch_sampler)
		sample_indices = next(sample_iter)
		data, targets = default_collate([self.dataset[i] for i in sample_indices])
		
		input_data, mask_label = self.gen_label(data,targets)

		return input_data, mask_label


class label_creator_from_gt(object):
	def __init__(self, m_labels):
		self.m_labels = m_labels

	def __call__(self,*input):
		data = input[0]
		targets = input[1]
		
		assert 'mask_label' in targets.keys()

		m_slice_idx = data.shape[1]//2
		mask_label = targets['mask_label']
		_input = []
		_input.append(data)
		for k,v in targets.iteritems():
			if k in self.m_labels:
				_input.append(v)

		_input.append(torch.unsqueeze(mask_label[:,m_slice_idx,:,:],1))
		input_data = torch.cat(_input,1)

		return input_data,mask_label




class label_creator_from_NNmodel(object):
	def __init__(self, nn_model, use_gpu, m_labels, watersed_cutoff):
		'''
		nn_model: Trained model from first network
		use_gpu: if True, use gpu to get prediction from first network
		m_labels: a list contains keys of layers to be included in the final input
		watershed_cutoff: cut off point of the water shed function
		'''
		self.model = nn_model
		self.use_gpu = use_gpu
		self.m_labels = m_labels
		self.cutoff = watersed_cutoff

	def __call__(self, *input):
		data = input[0]
		targets = input[1]

		assert 'seg_label' in targets.keys()

		self.model.eval()

		data   = Variable(data).float()
		# target = self.make_variable(targets)

		if self.use_gpu:
			data = data.cuda().float()
			# targets  = self.make_cuda_data(targets)

		## go through the first network
		preds  = self.model(data)
		# d = preds['distance'].data.cpu().numpy()
		# plt.imshow(d[0,0],cmap='gray')
		# plt.show()
		print(preds.keys())
		## apply water shed algorithmn to generate objects
		segments = self.segmentor(preds)
		print(segments.shape)
		## random choose one object, match the ground truth label to generate mask 
		seg_label = targets['seg_label']
		mask_in, mask_gt = self.mask_generator(segments, seg_label)
		# print('mask in shape is {} type is {}').format(mask_in.shape, type(mask_in))
		# print('mask ft shape is {} type is {}').format(mask_gt.shape, type(mask_gt))
		input_data       = self.gen_train_input(data,preds,mask_in)

		return input_data, mask_gt

	def segmentor(self, predict_labels):
		if 'final' in predict_labels.keys():
			dist_pred = predict_labels['final']
		else:
			dist_pred = predict_labels['distance']

		# print(dist_pred.size())
		# print(isinstance(dist_pred,Variable))
		segments = np.zeros(dist_pred.size())
		for i in range(dist_pred.size(0)):
			distance = dist_pred[i]
			# print('distance minimum is {}').format(np.min(distance.data.cpu().numpy()))
			# print('distance max is {}').format(np.max(distance.data.cpu().numpy()))

			segments[i] = self.watershed_d(distance, self.cutoff)

		return segments

	def mask_generator(self, seg, label):
		## Make sure that we have more than one slice in label
		assert label.size(1) > 1
		if isinstance(label, Variable):
			label = label.data

		label = label.cpu().numpy()
		mask_in = np.zeros(seg.shape)
		mask_gt = np.zeros(label.shape)
		for i in range(seg.shape[0]):
			_label = label[i]
			_seg = seg[i]

			_seg = np.squeeze(_seg)
			middle_label = _label[_label.shape[0]//2,:,:]
			s_id = np.unique(_seg).tolist()
			print('number of unique obj in predicted segment: {}').format(len(s_id))
			obj_id = np.random.choice(s_id, 1)
			_mask = (_seg == obj_id).astype(int)
			# u,c = np.unique(_mask, return_counts = True)
			# print(u)
			# print(c)

			middle_label_mask = np.multiply(middle_label, _mask)
			unique,count = np.unique(middle_label_mask[np.nonzero(middle_label_mask)], return_counts = True)
			obj_id_gt = unique[np.argmax(count)]

			mask_gt[i] = (_label == obj_id_gt).astype(int)
			mask_in[i] = np.expand_dims(_mask,0)

		mask_in = torch.from_numpy(mask_in).float()
		mask_gt = torch.from_numpy(mask_gt).float()
		return mask_in, mask_gt

	def gen_train_input(self,im_data,predict_labels,mask_in):
		if isinstance(im_data, Variable):
			im_data = im_data.data
		
		im_data = im_data.cpu()
		# print('im_data shape is {}, type is {}').format(im_data.shape,type(im_data))
		_input = []
		_input.append(im_data)
		for k,v in predict_labels.iteritems():
			if k in self.m_labels:
				# print('{} shape is {}, type is {}').format(k,v.data.shape,type(v.data.cpu()))
				_input.append(v.data.cpu())

		_input.append(mask_in)
		input_data = torch.cat(_input,1)
		return input_data

	def watershed_d(self,dist, cut_off = 3.5):
		if isinstance(dist, Variable):
			dist = dist.data
		
		dist = dist.cpu().numpy()
		dist =np.squeeze(dist)
		markers = dist > cut_off
		markers = skimage.morphology.label(markers)
		labels = watershed(-dist, markers)
		labels = np.expand_dims(labels,0)

		return labels




def test_label_gt():
	dataset = MaskDataset(	data_config = '../conf/cremi_datasets_with_tflabels.toml',
							out_patch_size = (224,224,3), 
							phase='valid', 
							transform = None,
							sub_dataset = 'Set_A', 
							subtract_mean=False)
	
	i_m_provider = instance_mask_provider(label_creator_from_gt(m_labels=['affinity','gradient','distance']),dataset,batch_size=3)
	input_data, mask_label = i_m_provider.generate()
	print('input data shape is {}').format(input_data.size())
	print('mask label shape is {}').format(mask_label.size())

	for i in range(input_data.size(0)):
		im = input_data[i,1].numpy()
		dist = input_data[i,3].numpy()
		grad_x = input_data[i,4].numpy()
		# grad_y = input_data[i,5].numpy()
		affinity = input_data[i,6].numpy()
		mask_in = input_data[i,7].numpy()

		lab_m = mask_label[i,1].numpy()

		fig,axes = plt.subplots(nrows =2, ncols=3,gridspec_kw = {'wspace':0.01, 'hspace':0.01})
		axes[0,0].imshow(im,cmap='gray')
		axes[0,0].axis('off')
		axes[0,0].margins(0,0)

		axes[0,1].imshow(dist)
		axes[0,1].axis('off')
		axes[0,1].margins(0,0)

		axes[0,2].imshow(affinity)
		axes[0,2].axis('off')
		axes[0,2].margins(0,0)

		axes[1,0].imshow(mask_in)
		axes[1,0].axis('off')
		axes[1,0].margins(0,0)

		axes[1,1].imshow(grad_x)
		axes[1,1].axis('off')
		axes[1,1].margins(0,0)

		axes[1,2].imshow(lab_m)
		axes[1,2].axis('off')
		axes[1,2].margins(0,0)

		plt.margins(x=0.001,y=0.001)
		plt.subplots_adjust(wspace=0, hspace=0)

		plt.show()
		plt.close('all')



if __name__ == '__main__':
	test_label_gt()

