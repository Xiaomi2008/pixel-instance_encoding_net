import h5py
from matplotlib import pyplot as plt 
from skimage.filters import gaussian
import numpy as np
from utils.utils import watershed_seg
from skimage.segmentation import relabel_sequential
from utils.evaluation import adapted_rand
from misc.orig_cremi_evaluation import voi


file = 'tempdata/resNet3D_predict_Set.h5'

F = h5py.File(file)
print(F.keys())


gs=lambda x: gaussian(x / float(np.max(x)), sigma=0.6, mode='reflect')
ts=lambda x, y: (x>y).astype(int)

predict_data_keys=[u'Set_A_affinityX', u'Set_A_affinityY', u'Set_A_affinityZ', \
                  u'Set_A_distance2D', u'Set_A_distance3D', 
                  u'Set_B_affinityX',  u'Set_B_affinityY', u'Set_B_affinityZ', \
                  u'Set_B_distance2D', u'Set_B_distance3D', \
                  u'Set_C_affinityX', u'Set_C_affinityY', u'Set_C_affinityZ', \
                  u'Set_C_distance2D', u'Set_C_distance3D']


def read_raw_image_data(set_name,subset_name):
	data_path ={}
	data_path['test'] ={'Set_A':'data/sample_A+_20160601.hdf',
	                    'Set_B':'data/sample_B+_20160601.hdf',
	                    'Set_C':'data/sample_C+_20160601.hdf'}

	data_path['valid'] ={'Set_A':'data/sample_A_20160501.hdf',
	                    'Set_B':'data/sample_B_20160501.hdf',
	                    'Set_C':'data/sample_C_20160501.hdf'}

	hd5_file = data_path[set_name][subset_name]
	h5=h5py.File(hd5_file)
	raw_h5_path=['volumes/raw']
	lb_h5_path =['volumes/labels']

	raw_im = h5[raw_h5_path]
	lbs    = h5[lb_h5_path]

	return raw_im, lbs

def show_merge_affinityXY(h5_dict, thick_idx, s_idx, d_set = 'A', gaus_f=False, threshold = None, sec_axis =0):
	x_idx ={'A':0,'B':5,'C':10}
	y_idx ={'A':1,'B':6,'C':11}
	X = h5_dict[h5_dict.keys()[x_idx[d_set]]]
	Y = h5_dict[h5_dict.keys()[y_idx[d_set]]]

	if sec_axis ==0:
		merge_XY= (X[thick_idx,s_idx] + Y[thick_idx,s_idx])/2.0
	elif sec_axis ==1:
		merge_XY= (X[thick_idx,:,s_idx,:] + Y[thick_idx,:,s_idx,:])/2.0
	elif sec_axis ==2:
		merge_XY= (X[thick_idx,:,:,s_idx] + Y[thick_idx,:,:,s_idx])/2.0

	merge_XY= gs(merge_XY) if gaus_f else merge_XY
	merge_XY =ts(merge_XY,threshold) if threshold else merge_XY
	fig = plt.figure(figsize=(800/96, 800/96), dpi=96)
	plt.imshow(merge_XY)
	plt.show(block = False)




def run_ws(set_name):
	dist = np.array(F[set_name])
	ws=watershed_dist(dist[0,0:10])
	seg_ws, _ , _ =relabel_sequential(ws)
	seg_ws=np.random.permutation(seg_ws.max() + 1)[seg_ws]
	return seg_ws

def watershed_dist(dist,threshold =0.06):
	return watershed_seg(dist, threshold)

def show_seg(vol,slice_idx):
	fig = plt.figure(figsize=(800/96, 800/96), dpi=96)
	
	plt.imshow(vol[slice_idx],cmap='spectral')
	plt.show(block=False)

def show_im_dict(h5_dict,dict_idx, thick_idx,s_idx,gaus_f =False,threshold =None, sec_axis =0):
     D = h5_dict[h5_dict.keys()[dict_idx]]
     show_im(D,thick_idx,s_idx,gaus_f,threshold, sec_axis)

def show_im(data_vol, thick_idx, s_idx,gaus_f =False,threshold =None, sec_axis =0):
	fig = plt.figure(figsize=(800/96, 800/96), dpi=96)
	if sec_axis ==0:
		D = data_vol[thick_idx, s_idx]
	elif sec_axis ==1:
		D = data_vol[thick_idx, :,s_idx,:]
	elif sec_axis ==2 :
		D = data_vol[thick_idx,:,:,s_idx]
	D = gs(D) if gaus_f else D
	D = ts(D,threshold) if threshold else D
	plt.imshow(D)
	plt.show(block = False)