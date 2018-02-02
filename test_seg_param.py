import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb
import h5py
from utils.evaluation import adapted_rand
from misc.orig_cremi_evaluation import voi
import torch
from utils.utils import watershed_seg, watershed_seg2

def get_data(hf5,set_name = 'Set_A'):
	# d_orig = hf5['d1_'+set_name]
	# d_combine =hf5['d2_'+set_name]
	# tg        =hf5['t1_'+set_name]
	d_orig = hf5[set_name +'_d1']
	d_combine =hf5[set_name +'_d2']
	tg        =hf5[set_name +'_t1']
	return d_orig, d_combine, tg



if __name__ == '__main__':
	hf5 = h5py.File('tempdata/seg_final_plus_distance.h5','r')
	#hf5 = h5py.File('tempdata/seg_fina_distance_only.h5','r')
	#hf5 =h5py.File('tempdata/seg_mu1_distance.h5', 'r')
	dataset = 'Set_A'
	#dataset = 'A'
	d_orig,d_combine,tg = get_data(hf5,dataset)
	
	t    = tg[100:,:,:]
	thresholds = np.linspace(16,35,15)
	arands = []
	print ('test {}'.format(dataset))
	for th in thresholds:
		#d_seg= watershed_seg2(d_orig[100:,:,:], d_combine[100:,:,:], threshold = th)
		d_seg= watershed_seg(d_combine[100:,:,:], threshold = th)
		#d_seg= watershed_seg(d_orig[100:,:,:], threshold = th)
		arand = adapted_rand(d_seg.astype(np.int), t)
		split, merge = voi(d_seg.astype(np.int), t)
		arands.append(arand)
		print('arand, split, merge = {:.3f}, {:.3f}, {:.3f} for threshold = {:.3f}'.format(arand,split,merge,th))
		#print('arand ={}  for threshold= {}'.format(arand,th))
	plt.plot(arands)
	plt.title('Set_' + dataset)
	plt.show()



