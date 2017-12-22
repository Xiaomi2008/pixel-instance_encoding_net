import os, sys
sys.path.append('../')
import pdb
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from utils.EMDataset import labelGenerator
from utils.EMDataset import CRIME_Dataset
from torch.utils.data import DataLoader, Dataloaderiter


class Proc_DataLoaderIter(Dataloaderiter):
	def __init__(self,loader,label_generator):
		super(Proc_DataLoaderIter,self).__init__(loader)
		self.label_generator =  label_generator
	def __make_input_data__(self,data,preds):
	 	'''we can concate preds into data as etra channels
           for noew, we just return data
	 	 '''
	 	 return data

class NN_proc_DataLoaderIter(DataLoaderIter):
	def __init__(self,loader,nn_model):
		super(NN_proc_DataLoaderIter,self).__init__(loader)
		self.nn_model = nn_model
	def __next__(self):
		batch=super(NN_proc_DataLoaderIter,self).__next__()
		data,targets,seg_label = batch
		preds           = self.nn_model(data)
		pred_seg        = self.__segment__(preds)
		Mask_in,Mask_gt = self.__make_mask__(pred_seg,seg_label)
		input_data      = self.__make_input_data__(data,preds)
		return input_data, Mask_in, Mask_gt

	def __segment__(self,preds):
		distance = preds['distance']
		assert distance.dim ==4 and distance[1] ==1
		seg2D_list = [ watershed_seg2D(torch.squeeze(distance[i]))
		               for i in range(len(distnce.shape[0]))]
		seg_batch  =  torch.stack( 
				       			  [  torch.usqueeze(torch.Tensor(seg2d),0)
				       			     for seg2d in seg2d_list
				       			  ],
				       			  dim =0
			       			     )
		return seg_batch
	
	def __make_mask__(self,pred_seg_batch,target_seg_batch):
		n_samples = pred_seg_batch.shape[0]
		unqiue_ids_in_each_seg      = [np.unqiue(
			                                     pred_seg_batch[i].cpu.numpy()
			                                    ) 
		                                for i in range(n_samples)
		                              ]

		selected_seg_ids            =  [np.random.choice(unqiue_ids_in_each_seg[i]) 
		                                for i in range(n_samples)
		                               ]
		masked_preds                =  [(pred_seg_batch[i] == selected_seg_ids[i]).astype(np.int) 
		                                for i in range(n_samples)]

		masked_target_ids           =  [find_max_coverage_id(masked_preds[i], seg[i,seg[i].shape[0]//2]) 
		                                for i in range(n_samples)]
		
		masked_target               =   [ (target_seg_batch[i]== masked_target_ids[i]).astype(np.int) 
		                                for i in range(n_samples)]

	    maksed_target_batch =  np.concatenate(masked_target,dim = 0)
	    maksed_preds_batch  =  np.concatenate(masked_preds,dim =0)
	    return maksed_preds_batch, maksed_target_batch

	 


class GT_proc_DataLoaderIter(DataLoaderIter):
	def __init__(self,loader):
		super(GT_proc_DataLoaderIter,self).__init__(loader)
	def __next__(self):
		batch=super(GT_proc_DataLoaderIter,self).__next__()
		data,targets,seg_label = batch
		Mask_in,Mask_gt = self.__make_mask__(seg_label)
		input_data      =  self.__make_input_data__(data,targets)
		return input_data, Mask_in, Mask_gt

	def __make_mask__(self,target_seg_batch):
		n_samples = target_seg_batch.shape[0]
		unqiue_ids_in_each_seg      = [np.unqiue(
			                                     target_seg_batch[i].cpu.numpy()
			                                    ) 
		                                for i in range(n_samples)
		                              ]

		selected_seg_ids            =  [np.random.choice(unqiue_ids_in_each_seg[i]) 
		                                for i in range(n_samples)
		                               ]
		
		masked_target               =   [ (target_seg_batch[i]== selected_seg_ids[i]).astype(np.int) 
		                                for i in range(n_samples)]

		masked_preds                = [ np.expand_dims(maksed_target[i, maksed_target[i].shape[0]//2],0)
		                                for i in range(n_samples)]               

	    maksed_target_batch =  np.concatenate(masked_target,dim = 0)
	    maksed_preds_batch  =  np.concatenate(masked_preds,dim =0)
	    return maksed_preds_batch, maksed_target_batch


class instance_mask_Dataloader(DataLoader):
	def __init__(self,**kwargs):
		super(instance_mask_provider,self).__init__(kwargs)

	def __iter__(self):
		'''subclass should return different Dataloaderiter'''
		raise NotImplementedError(" __iter__ function must be implemented in subclass !")

	def gen_train_input(data,Preds,Mask_in):
		input_data = torch.cat([data,Preds['distance'], Mask_in],1)
		return input_data



class instance_mask_NNproc_Dataloader(instance_mask_Dataloader):
	def __init__(self,nn_model,**kwargs):
		super(instance_mask_NNproc_Dataloader,self).__init__(kwargs)
		self.nn_model = nn_model
	def __iter__(self):
        return NN_proc_DataLoaderIter(self,self.nn_model)



class instance_mask_GTproc_DataLoader(instance_mask_Dataloader):
	def __init__(self,**kwargs):
		super(instance_mask_GTproc_DataLoader,self).__init__(kwargs)
		#self.CRIME_Dataset_3D = CRIME_Dataset_3D_labels(kwargs)
		#self.data_loader = DataLoader
	def __iter__(self):
        return GT_proc_DataLoaderIter(self,self.nn_model)

class CRIME_Dataset_3D_labels(CRIME_Dataset):
	def __init__(self,
                 out_patch_size       =   (224,224,1), 
                 sub_dataset          =   'Set_A',
                 subtract_mean        =   True,
                 phase                =   'train',
                 transform            =   None,
                 data_config          =   'conf/cremi_datasets_with_tflabels.toml'):
	    super(CRIME_Dataset_with_NNgen_labels,self).__init__(sub_dataset=sub_dataset, 
                                         out_patch_size =out_patch_size,
                                         subtract_mean =subtract_mean,
                                         phase = phase,
                                         transform =transform)

	def __getitem__(self, index):
        im_data,lb_data= self.random_choice_dataset(self.im_lb_pair)
        data,seg_label = self.get_random_patch(index,im_data,lb_data)
        '''Convert seg_label to 2D by obtaining only intermedia slice 
           while the input data have multiple slice as multi-channel input
           the network only output the prediction of sigle slice in the center of Z dim'''
        tc_data        = torch.from_numpy(data).float()
        tc_label_dict  = self.gen_label_per_slice(seg_label)
        return tc_data, tc_label_dict, seg_label
    
    def gen_label_per_slice(self,seg):
    	if seg.ndim == 3:
    		z_dim = seg.shapep[0]
    		assert ((z_dim % 2) == 1) #  need ensure that # slices is odd number
    		slice_seg_list = [slice_seg_lb[i,:,:] for i in range(z_dim)]
    		slice_tgDict_list  = self.label_generator(*slice_seg_list)

    		lb_dict= {}
    		for k in slice_seg_list[0].keys():
    			#V = [ slice_seg_list[i][k] for i in range(len(slice_seg_list))]
    			lb_dict[k] = torch.cat([slice_seg_list[i][k] for i in range(len(slice_seg_list))],dim=0)

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
    #hat = ndimage.black_tophat(distance, 14)
    # Combine with denoised image
    #hat -= 0.3 * distance
    # Morphological dilation to try to remove some holes in hat image
    #hat = skimage.morphology.dilation(hat)
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
	bool_mask = mask.astype(np.bool)
	converted_ids=seg[bool_mask]
	unqiue_ids,count = np.unique(converted_ids,return_count = True)
	idex = np.argmax(count)
	return unqiue_ids[idex]


def test():
	dataset = CRIME_Dataset_3D_labels(out_patch_size       =   (224,224,3), 
						              sub_dataset          =   'ALL',
						              subtract_mean        =   True,
						              phase                =   'train',
						              transform            =   None,
						              data_config          =   '../conf/cremi_datasets_with_tflabels.toml')
	GT_Mask_DataLoader = instance_mask_GTproc_DataLoader(dataset    = dataset,
                                						batch_size  = 10,
                                						shuffle     = True,
                                						num_workers = 1)
	for i, (data,targets) in enumerate(GT_Mask_DataLoader, 0):
		if i > 10:
			break

if __name__ == '__main__':
  test()


