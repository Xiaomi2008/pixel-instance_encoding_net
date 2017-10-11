from scipy.ndimage.morphology import distance_transform_edt as dis_transform
import scipy.ndimage as nd
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import pdb
import multiprocessing as mp
from multiprocessing import Array
import ctypes
from volumes import HDF5Volume

training_data_config_file = '../conf/cremi_datasets.toml'
cremi_volumes = HDF5Volume.from_toml(training_data_config_file)
label_ids = cremi_volumes[cremi_volumes.keys()[0]]
lb_array = np.array(label_ids).astype(np.float32)
ids = np.unique(lb_array).tolist()
dt_map_y = np.zeros(lb_array.shape)
# dt_map_i = np.zeros([lb_array.ndim] + list(lb_array.shape))
dir_map  = np.zeros([lb_array.ndim+1] + list(lb_array.shape)).astype(np.float32)

lb_shape = lb_array.shape
dirmap_shape = dir_map.shape

dir_map.shape =1*dir_map.size
lb_array.shape=1*lb_array.size
shared_dirmap_arry = Array(ctypes.c_float, dir_map)
share_lb_array = Array(ctypes.c_float, lb_array)
def gradient_worker(n_id):
	print("dfddfd")
	print("size of {} of type {}".format(len(n_id),type(n_id)))
	n_ids = n_id if isinstance(n_id,list) else [n_id]
	sum_gx = np.zeros(lb_shape)
	sum_gy = np.zeros(lb_shape)
	sum_gz = np.zeros(lb_shape)
	sum_dy = np.zeros(lb_shape)
	shared_lb = np.frombuffer(share_lb_array.get_obj(),dtype=np.float32).reshape(lb_shape)
	print("done creating zeros array")
	#pdb.set_trace()

	for obj_id in n_ids:
		obj_array = (shared_lb == obj_id).astype(int)
		#pdb.set_trace()
		dt_y=  dis_transform(obj_array)
		dx,dy,dz =1,1,1
		gx,gy,gz=np.gradient(dt_y,dx,dy,dz,edge_order =1)
		gx-=np.min(gx)+0.01
		gy-=np.min(gy)+0.01
		gz-=np.min(gz)+0.01
		sum_gx+=gx
		sum_gy+=gy
		sum_gz+=gz
		sum_dy+=dt_y
	with shared_dirmap_arry.get_lock():
		shared_num_dirmap = np.frombuffer(shared_dirmap_arry.get_obj(),dtype=np.float32).reshape(dirmap_shape)
		shared_num_dirmap[0,:,:]+=sum_gx
		shared_num_dirmap[1,:,:]+=sum_gy
		shared_num_dirmap[2,:,:]+=sum_gz
		shared_num_dirmap[3,:,:]+=sum_dy
	return n_id
def update_callback(result):
	print("callback finished with {} out of {}".format(n_id,))
def start_parallel_gradient():
	num_process = 10
	pool = mp.Pool(processes=num_process)
	for i in range(num_process):
		steps = len(ids) /num_process
		if i <9:
			n_ids = ids[i*steps:(i+1)*steps]
		else:
			n_ids =ids[i*steps:]
		print("apply async on {} ".format(len(n_ids)))
		pool.apply_async(gradient_worker, args=(n_ids,), callback = update_callback)
	pool.close()
	pool.join()

if __name__ =='__main__':
	start_parallel_gradient()
	shared_num_dirmap = np.frombuffer(shared_dirmap_arry.get_obj(),dtype=np.float32).reshape(dirmap_shape)
	plt.imshow(shared_num_dirmap[0,1,:,:])
	plt.savefig('dir_map_1.png')
	plt.imshow(shared_num_dirmap[1,1,:,:])
	plt.savefig('dir_map_2.png')
	#plt.imshow(shared_num_dirmap[1,:,:])
	#plt.imshow(dir_map[1,:,:,:300])
	#plt.savefig('dir_map_3.png')
	#tr_im =  dis_transform(label_ids)