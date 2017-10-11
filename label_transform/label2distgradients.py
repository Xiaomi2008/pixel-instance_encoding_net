from scipy.ndimage.morphology import distance_transform_edt as dis_transform
import scipy.ndimage as nd
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import pdb
import multiprocessing as mp
#from multiprocessing import Array
import ctypes
from volumes import HDF5Volume



data_config = '../conf/cremi_datasets.toml'
volumes = HDF5Volume.from_toml(data_config)
v_names = volumes.keys()





def update_callback(rs_return):
		n_id = rs_return
		print("callback finished {} out of {}".format(n_id,len(obj_ids)))
def gradient_worker_2D_on_xy(slices):
		shared_lb = np.frombuffer(share_lbs.get_obj(),dtype=np.float32).reshape(lb_shape)
		print (slices)
		dirmap = np.frombuffer(shared_dirmap.get_obj(),dtype=np.float32).reshape(dirmap_shape)
		for slice_id in slices:
			#print('start slice {} in xy'.format(slice_id) )
			assert slice_id >=0 and slice_id < lb_shape[0]
			lb_xy_shape = lb_shape[1],lb_shape[2]
			sum_gx = np.zeros(lb_xy_shape)
			sum_gy = np.zeros(lb_xy_shape)
			sum_gz = np.zeros(lb_xy_shape)
			sum_dt = np.zeros(lb_xy_shape)
			slice_lbs =shared_lb[slice_id,:,:]
			s_ids =np.unique(slice_lbs).tolist()
			#print('there are {} objects in slice {}'.format(len(s_ids),slice_id))
			for obj_id in s_ids:
				obj_arr = (slice_lbs == obj_id).astype(int)
				dt	=  dis_transform(obj_arr)
				dx,dy 	= 1,1
				gx,gy   = np.gradient(dt,dx,dy,edge_order =1)
				gx-=np.min(gx)+0.01
				gy-=np.min(gy)+0.01
				sum_gx+=gx
				sum_gy+=gy
				sum_dt+=dt

			with shared_dirmap.get_lock():
				dirmap[1,slice_id,:,:]+=sum_gx
				dirmap[2,slice_id,:,:]+=sum_gy
				# dimension 4 stores the distance transform
				dirmap[3,slice_id,:,:]+=sum_dt
		return slices

def gradient_worker_2D_on_z(slices):
		shared_lb = np.frombuffer(share_lbs.get_obj(),dtype=np.float32).reshape(lb_shape)
		dirmap = np.frombuffer(shared_dirmap.get_obj(),dtype=np.float32).reshape(dirmap_shape)
		for slice_id in slices:
			assert slice_id >=0 and slice_id < lb_shape[2]
			# this shape supporse to be 125 by 1250
			lb_xy_shape = lb_shape[0],lb_shape[2]
			sum_gx = np.zeros(lb_xy_shape)
			sum_gz = np.zeros(lb_xy_shape)
			sum_dt = np.zeros(lb_xy_shape)
			slice_lbs =shared_lb[:,:,slice_id]
			s_ids =np.unique(slice_lbs).tolist()
			for obj_id in s_ids:
				obj_arr = (slice_lbs == obj_id).astype(int)
				dt	=  dis_transform(obj_arr)
				dx,dz 	= 1,1
				gz,gx   = np.gradient(dt,dx,dz,edge_order =1)
				gz-=np.min(gz)+0.01
				sum_gz+=gz
				sum_dt+=dt
			with shared_dirmap.get_lock():
				dirmap[0,:,:,slice_id]+=sum_gz
				# dimension 4 stores the distance transform
				dirmap[3,slice_id,:,:]+=sum_dt
		return slices
def gradient_worker_3D(n_id):
		print("start creating zeros array")
		n_ids = n_id if isinstance(n_id,list) else [n_id]
		sum_gx = np.zeros(lb_shape)
		sum_gy = np.zeros(lb_shape)
		sum_gz = np.zeros(lb_shape)
		sum_dy = np.zeros(lb_shape)
		shared_lb = np.frombuffer(share_lbs.get_obj(),dtype=np.float32).reshape(lb_shape)
		print("done creating zeros array")

		for obj_id in n_ids:
			obj_array = (shared_lb == obj_id).astype(int)
			#pdb.set_trace()
			dt_y=  dis_transform(obj_array)
			dx,dy,dz =1,1,1
			gz,gx,gy=np.gradient(dt_y,dx,dy,dz,edge_order =1)
			gx-=np.min(gx)+0.01
			gy-=np.min(gy)+0.01
			gz-=np.min(gz)+0.01
			sum_gx+=gx
			sum_gy+=gy
			sum_gz+=gz
			sum_dy+=dt_y
		with shared_dirmap.get_lock():
			dirmap = np.frombuffer(shared_dirmap.get_obj(),dtype=np.float32).reshape(dirmap_shape)
			dirmap[0,:,:]+=sum_gz
			dirmap[1,:,:]+=sum_gx
			dirmap[2,:,:]+=sum_gy
			# dimension 4 stores the distance transform
			dirmap[3,:,:]+=sum_dy
		return n_id
def start(mode='2D'):
		'''
		 Funciton:
		 	Start multiprocessing  to fill a shared memoery (multiprocess Array)
		 	with results of direction map (gradient) and distance transform.

		 Input:
		 	model option : 3D, 2D
		'''

		print('start mode = {}'.format(mode))
		num_process = 10
		pool = mp.Pool(processes=num_process)
		for i in range(num_process):
			if mode == '3D':
				steps = len(obj_ids) /num_process
				n_ids = obj_ids[i*steps:(i+1)*steps] if i <num_process-1 else obj_ids[i*steps:]
				pool.apply_async(gradient_worker_3D, args=(n_ids,), callback = update_callback)
			elif mode == '2D':
				num_slices_z_axis = lb_shape[0]
				num_slices_y_axis = lb_shape[2]
				steps_z = num_slices_z_axis / num_process
				slices_z_ids = [sid for sid in range(steps_z*i,steps_z*(i+1))] if i < num_process-1 \
							 else  [sid for sid in range(steps_z*i,num_slices_z_axis)]
				pool.apply_async(gradient_worker_2D_on_xy, args=(slices_z_ids,),callback = update_callback)

				steps_y = num_slices_y_axis / num_process
				slices_y_ids = [sid for sid in range(steps_y*i,steps_y*(i+1))] if i < num_process-1 \
							 else  [sid for sid in range(steps_y*i,num_slices_y_axis)]
				pool.apply_async(gradient_worker_2D_on_z, args=(slices_y_ids,), callback = update_callback)
				#print('completed start 2D process {}'.format(i))
		pool.close()
		pool.join()
	
def test():
	start(mode='2D')
	dirmap = np.frombuffer(shared_dirmap.get_obj(),dtype=np.float32).reshape(dirmap_shape)
	plt.imshow(dirmap[0,3,:,:])
	plt.savefig('dir_map_z.png')
	plt.imshow(dirmap[1,3,:,:])
	plt.savefig('dir_map_x.png')
	plt.imshow(dirmap[2,3,:,:])
	plt.savefig('dir_map_y.png')
	plt.imshow(dirmap[3,3,:,:])
	plt.savefig('dist_map.png')
if __name__ =='__main__':
	volume_names =  volumes.keys()
	for v_name in volume_names:
		lb_array = volumes[v_name].label_data
		lb_shape = lb_array.shape
		print (v_name)
		lb_array = np.array(lb_array).astype(np.float32)
		obj_ids  = np.unique(lb_array).tolist()
		# ----  number of directional maps =  gradient along each of its demention + 1 distance transform map ----
		dir_map  = np.zeros([lb_array.ndim+1] + list(lb_array.shape)).astype(np.float32)
		dirmap_shape = dir_map.shape
		dir_map.shape =1*dir_map.size
		lb_array.shape=1*lb_array.size
		shared_dirmap = mp.Array(ctypes.c_float, dir_map)
		share_lbs = mp.Array(ctypes.c_float, lb_array)