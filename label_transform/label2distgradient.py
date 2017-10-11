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

def update_callback(lb_process_obj,rs_return):
		n_id = rs_return
		print("callback finished {} out of {}".format(n_id,len(self.obj_ids)))
def gradient_worker_2D_on_xy(lb_process_obj,slices):
		print('start slice {} in xy')
		shared_lb = np.frombuffer(lb_process_obj.share_lbs.get_obj(),dtype=np.float32).reshape(lb_process_obj.lb_shape)
		print('done form shared lb')
		for slice_id in slices:
			print('start slice {} in xy'.format(slice_id) )
			assert slice_id >=0 and slice_id < lb_process_obj.lb_shape[0]
			lb_xy_shape = lb_process_obj.lb_shape[1],lb_process_obj.lb_shape[2]
			sum_gx = np.zeros(lb_xy_shape)
			sum_gy = np.zeros(lb_xy_shape)
			sum_gz = np.zeros(lb_xy_shape)
			sum_dt = np.zeros(lb_xy_shape)
			slice_lbs =shared_lb[slice_id,:,:]
			s_ids =np.unique(slice_lbs).tolist()
			print('start 2 slice {} in xy'.format(slice_id) )
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
			with lb_process_obj.shared_dirmap.get_lock():
				dirmap = np.frombuffer(lb_process_obj.shared_dirmap.get_obj(),dtype=np.float32).reshape(lb_process_obj.dirmap_shape)
				dirmap[0,slice_id,:,:]+=sum_gx
				dirmap[1,slice_id,:,:]+=sum_gy
				# dimension 4 stores the distance transform
				dirmap[3,slice_id,:,:]+=sum_dt
			return slices

class label_preprocessor:
	def __init__(self, label_array):
		self.lb_shape = label_array.shape
		self.lb_array = np.array(label_array).astype(np.float32)
		self.obj_ids  = np.unique(self.lb_array).tolist()
		# ----  number of directional maps =  gradient along each of its demention + 1 distance transform map ----
		self.dir_map  = np.zeros([self.lb_array.ndim+1] + list(self.lb_array.shape)).astype(np.float32)
		self.dirmap_shape = self.dir_map.shape
		self.dir_map.shape =1*self.dir_map.size
		self.lb_array.shape=1*self.lb_array.size
		self.shared_dirmap = mp.Array(ctypes.c_float, self.dir_map)
		self.share_lbs = mp.Array(ctypes.c_float, self.lb_array)
	def gradient_worker_2D_on_xy(self,slices):
		print('start slice {} in xy')
		shared_lb = np.frombuffer(self.share_lbs.get_obj(),dtype=np.float32).reshape(self.lb_shape)
		print('done form shared lb')
		for slice_id in slices:
			print('start slice {} in xy'.format(slice_id) )
			assert slice_id >=0 and slice_id < self.lb_shape[0]
			lb_xy_shape = self.lb_shape[1],self.lb_shape[2]
			sum_gx = np.zeros(lb_xy_shape)
			sum_gy = np.zeros(lb_xy_shape)
			sum_gz = np.zeros(lb_xy_shape)
			sum_dt = np.zeros(lb_xy_shape)
			slice_lbs =shared_lb[slice_id,:,:]
			s_ids =np.unique(slice_lbs).tolist()
			print('start 2 slice {} in xy'.format(slice_id) )
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
			with self.shared_dirmap.get_lock():
				dirmap = np.frombuffer(self.shared_dirmap.get_obj(),dtype=np.float32).reshape(self.dirmap_shape)
				dirmap[0,slice_id,:,:]+=sum_gx
				dirmap[1,slice_id,:,:]+=sum_gy
				# dimension 4 stores the distance transform
				dirmap[3,slice_id,:,:]+=sum_dt

	def gradient_worker_2D_on_z(self,slices):
		shared_lb = np.frombuffer(self.share_lbs.get_obj(),dtype=np.float32).reshape(self.lb_shape)
		for slice_id in slices:
			assert slice_id >=0 and slice_id < self.lb_shape[2]
			# this shape supporse to be 125 by 1250
			lb_xy_shape = self.lb_shape[0],self.lb_shape[2]
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
			with self.shared_dirmap.get_lock():
				dirmap = np.frombuffer(self.shared_dirmap.get_obj(),dtype=np.float32).reshape(dirmap_shape)
				dirmap[0,:,:,slice_id]+=sum_gz
				# dimension 4 stores the distance transform
				dirmap[3,slice_id,:,:]+=sum_dt




	def gradient_worker_3D(self,n_id):
		print("start creating zeros array")
		n_ids = n_id if isinstance(n_id,list) else [n_id]
		sum_gx = np.zeros(self.lb_shape)
		sum_gy = np.zeros(self.lb_shape)
		sum_gz = np.zeros(self.lb_shape)
		sum_dy = np.zeros(self.lb_shape)
		shared_lb = np.frombuffer(self.share_lb_array.get_obj(),dtype=np.float32).reshape(self.lb_shape)
		print("done creating zeros array")

		for obj_id in n_ids:
			obj_array = (self.shared_lb == obj_id).astype(int)
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
		with self.shared_dirmap.get_lock():
			dirmap = np.frombuffer(self.shared_dirmap.get_obj(),dtype=np.float32).reshape(self.dirmap_shape)
			dirmap[0,:,:]+=sum_gz
			dirmap[1,:,:]+=sum_gx
			dirmap[2,:,:]+=sum_gy
			# dimension 4 stores the distance transform
			dirmap[3,:,:]+=sum_dy
		return n_id
	def update_callback(self,rs_return):
		n_id = rs_return
		print("callback finished {} out of {}".format(n_id,len(self.obj_ids)))
	def start(self,mode='3D'):
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
				steps = len(self.obj_ids) /num_process
				n_ids = self.obj_ids[i*steps:(i+1)*steps] if i <num_process-1 else self.obj_ids[i*steps:]
				print("apply async on {} ".format(len(n_ids)))
				pool.apply_async(self.gradient_worker_3D, args=(n_ids,), callback = update_callback)
			elif mode == '2D':
				num_slices_z_axis = self.lb_shape[0]
				num_slices_y_axis = self.lb_shape[2]
				steps_z = num_slices_z_axis / num_process
				#print('starting 2D process {}'.format(i))
				slices_z_ids = [sid for sid in range(steps_z*i,steps_z*(i+1))] if i < num_process-1 \
							 else  [sid for sid in range(steps_z*i,num_slices_z_axis)]
				print slices_z_ids
				#pool.apply_async(gradient_worker_2D_on_xy, args=(self,slices_z_ids,), callback = self.update_callback)
				pool.apply_async(gradient_worker_2D_on_xy, args=(self,slices_z_ids,),callback = update_callback)

				# steps_y = num_slices_y_axis / num_process
				# slices_y_ids = [sid for sid in range(steps_y*i,steps_y*(i+1))] if i < num_process-1 \
				# 			 else  [sid for sid in range(steps_y*i,num_slices_y_axis)]
				# pool.apply_async(self.gradient_worker_2D_on_z, args=(slices_y_ids,), callback = self.update_callback)
				#print('completed start 2D process {}'.format(i))
		pool.close()
		pool.join()
def test():
	data_config = '../conf/cremi_datasets.toml'
	volumes = HDF5Volume.from_toml(data_config)
	v_names = volumes.keys()
	lb_array = volumes[v_names[0]].label_data
	input_label = lb_array[0:10,:,:]
	LP= label_preprocessor(lb_array)
	LP.start(mode='2D')
	dirmap = np.frombuffer(LP.shared_dirmap.get_obj(),dtype=np.float32).reshape(LP.dirmap_shape)
	plt.imshow(dirmap[0,1,:,:])
	plt.savefig('dir_map_1.png')
	plt.imshow(dirmap[1,1,:,:])
	plt.savefig('dir_map_2.png')

if __name__ =='__main__':
	test()