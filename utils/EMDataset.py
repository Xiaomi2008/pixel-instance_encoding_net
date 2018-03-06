
import os, sys

sys.path.append('../')
import pdb
import torch
import numpy as np
from transform import *
# import cv2
import torch.nn.functional as F
from matplotlib import pyplot as plt
# import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader
from label_transform.volumes import Volume
from label_transform.volumes import HDF5Volume
from label_transform.volumes import bounds_generator
from label_transform.volumes import SubvolumeGenerator
from transform import RandomContrast
from scipy.interpolate import RegularGridInterpolator
from math import ceil
from utils import replace_bad_slice_in_test


class exp_Dataset(Dataset):
    """base dataset"""
    def __init__(self,
                 out_patch_size=(224, 224, 1),
                 in_patch_size=None,
                 sub_dataset='All',
                 subtract_mean=True,
                 phase='train',
                 channel_axis = 2,
                 transform=None,
                 label_config=None):
                 #label_gen=None):

        self.sub_dataset = sub_dataset
        self.phase = phase
        self.channel_axis = (channel_axis + 1) % 3

        #self.out_patch_size = out_patch_size

        self.x_out_size = out_patch_size[0]
        self.y_out_size = out_patch_size[1]
        self.z_out_size = out_patch_size[2]
        self.in_patch_size = in_patch_size if in_patch_size else out_patch_size
        self.x_in_size = self.in_patch_size[0]
        self.y_in_size = self.in_patch_size[1]
        self.z_in_size = self.z_out_size

        self.subtract_mean = subtract_mean
        self.transform = transform

        ''' subclass should assign this param befor calling  __getitm__ function '''
        self.slice_start_z = 0

        self.set_phase(phase)
        self.im_lb_pair = self.load_data()

        im_data = self.im_lb_pair[self.im_lb_pair.keys()[0]]['image']
        dim_shape = im_data.shape

        self.y_size = dim_shape[2] - self.x_in_size + 1
        self.x_size = dim_shape[1] - self.y_in_size + 1
        # self.y_size = dim_shape[2] - self.x_out_size + 1
        # self.x_size = dim_shape[1] - self.y_out_size + 1

        # self.label_generator  = label_transform(objSizeMap =True)
        #self.label_generator = label_gen if label_gen else labelGenerator()
        self.label_config = label_config
        self.label_generator = labelGenerator(self.label_config)

    def output_labels(self):
        return self.label_generator.output_labels()
        # return ['gradient','affinity','centermap','sizemap','distance']

    def __getitem__(self, index):

        # random choice one of sub_datasets
        im_data, lb_data = self.random_choice_dataset(self.im_lb_pair)
        data, seg_label = self.get_random_patch(index, im_data, lb_data)

        print('line 75 seg_label shape = {}'.format(seg_label.dim_shape))

        tc_data = torch.from_numpy(data).float()
        tc_label_dict = self.label_generator(seg_label)[0]
        return tc_data, tc_label_dict

    def random_choice_dataset(self, im_lb_pair):
        dataset_id = np.random.choice(im_lb_pair.keys())
        im_data = self.im_lb_pair[dataset_id]['image']
        lb_data = self.im_lb_pair[dataset_id]['label']
        return im_data, lb_data

    def get_random_patch(self, index, im_data, lb_data):
        #z_start = index // (self.x_size * self.y_size) + self.slice_start_z
        #remain = index % (self.x_size * self.y_size)
        #x_start = remain // self.y_size
        #y_start = remain % self.y_size
        #
        #random.seed(index)
        #
        #warp_mode = np.random.choice(['fixed_size','variant_size'])

        #warp_mode = 'fixed_size' #'variant_size'

        warp_mode = 'variant_size'
        #print (warp_mode)

        x_in_size = self.x_in_size
        y_in_size = self.y_in_size
        if warp_mode == 'variant_size' and self.phase =='train':
            x_scale_factor=np.random.choice(np.linspace(0.5,2.1, 32))
            y_scale_factor=np.random.choice(np.linspace(0.5,2.1, 32))
            x_in_size = int(ceil(self.x_in_size * x_scale_factor))
            y_in_size = int(ceil(self.y_in_size * y_scale_factor))


        im_z_size, im_x_size, im_y_size = im_data.shape[0], im_data.shape[1], im_data.shape[2]
        if im_z_size - self.slice_start_z -1 < self.z_out_size:
            z_start = self.slice_start_z
            z_end   = im_z_size
        else:
            z_start = np.random.randint(self.slice_start_z, self.slice_end_z - self.z_out_size -1)
            z_end   = z_start +  self.z_out_size
       

        x_start = index % (im_x_size - x_in_size - 1)
        y_start = (index // self.x_size) % (im_y_size - y_in_size - 1)

        if z_start > im_z_size - self.z_in_size:
            z_start = im_z_size - self.z_in_size

        x_end = x_start + x_in_size
        y_end = y_start + y_in_size

        x_start = 0 if x_start <0 else x_start
        y_start = 0 if y_start <0 else y_start

        x_end = im_x_size if x_end >im_x_size else x_end
        y_end = im_y_size if y_end >im_y_size else y_end
        


        # random slice the data from one of side direction if channel_axis id not in z_axis
        #transpose_side_slice = True if self.channel_axis >0 and np.random.randint(2)>0 else False
        transpose_side_slice =  False
        if transpose_side_slice:
           data = np.array(im_data[z_start:z_end, y_start:y_end, x_start:x_end]).astype(np.float)
           seg_label = np.array(lb_data[z_start:z_end, y_start:y_end, x_start:x_end]).astype(np.int)
        else:
           data = np.array(im_data[z_start:z_end, x_start:x_end, y_start:y_end]).astype(np.float)
           seg_label = np.array(lb_data[z_start:z_end, x_start:x_end, y_start:y_end]).astype(np.int)



        # random crop each slize in xy plane, so that we can augmented mis-aligment.
        # after crop it relies on the interpolatation to recover it size to fit the output size requirment
        if warp_mode == 'variant_size':
            mis_align_scale  =0.1
            d_x_size = data.shape[1]
            d_y_size = data.shape[2]
            x_offset = int(d_x_size *mis_align_scale)
            y_offset = int(d_y_size *mis_align_scale)
            d_new_x_size = d_x_size - x_offset
            d_new_y_size = d_y_size - y_offset
            new_d_list =[]
            new_seg_list=[]
            for j in range(len(data)): # length of z-direction
                 if x_offset==0:
                    print('d_size ={} x_start,xend={} : {}'.format(d_x_size,x_start,x_end))
                    pdb.set_trace()
                 if y_offset==0:
                    print('d_size ={} y_start,yend={} : {}'.format(d_y_size,y_start,y_end))
                    pdb.set_trace()
                 x_start = np.random.randint(0,x_offset)
                 y_start = np.random.randint(0,y_offset)
                 x_end   = x_start+d_new_x_size
                 y_end   = y_start+d_new_y_size
                 new_d_list.append(data[j, x_start:x_end,y_start:y_end])
                 new_seg_list.append(seg_label[j,x_start:x_end,y_start:y_end])
            # for i,nda in enumerate(new_d_list):
            #     print('s {} = {}'.format(i,nda.shape))
            data = np.stack(new_d_list,0)
            seg_label = np.stack(new_seg_list,0)

            x_in_size = d_new_x_size
            y_in_size = d_new_y_size

             #data=np.stack([d[i, x_start:x_end, y_start:y_end] for d in data],0)



        #print('before: data shape ={} seg_lable shape ={}'.format(data.shape,seg_label.shape))

        in_out_equal = (x_in_size == self.x_out_size) and (y_in_size ==self.y_out_size)

        if warp_mode == 'variant_size' and self.phase == 'train' or not in_out_equal:
            # assert x_in_size >2
            # assert y_in_size >2
            # assert self.z_in_size >1
            x = np.linspace(0,x_in_size-1, x_in_size)
            y = np.linspace(0,y_in_size-1, y_in_size)
            z = np.linspace(0,self.z_in_size-1, self.z_in_size)

            # assert len(x) >1
            # assert len(y) >1
            # assert len(z) >1



            '''Fix the dived by invalide data error  when USING interpolate in this "get_random_patch",
               which can lead the network explosion due to the NaN input to the networks
               
               your_python_path/lib/python2.7/site-packages/scipy/interpolate/interpolate.py
               Line 2468 or 2500 
               in Function _find_indices(self,xi):

               Change:
               norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
               To:
               norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]).clip(min=0.001)) '''

            data_interp = RegularGridInterpolator((z,x,y), data,fill_value=None, bounds_error=False)
            lb_interp = RegularGridInterpolator((z,x,y),seg_label,fill_value=None, bounds_error=False)


            xi = np.linspace(0,x_in_size-1,self.x_out_size)
            yi = np.linspace(0,y_in_size-1,self.y_out_size)
            zi = np.linspace(0,self.z_in_size-1,self.z_out_size)

            # assert len(xi) >1
            # assert len(yi) >1
            # assert len(zi) >1
           
            #a,b,c=np.meshgrid(zi,xi,yi, indexing='ij')
            #print('a {} b {} c{} '.format(a.shape,b.shape,c.shape))
            pts  = np.squeeze(np.array(zip(np.meshgrid(zi,xi,yi, indexing='ij'))))
            
            data = data_interp(pts.T, method ='linear')
            seg_label = lb_interp(pts.T, method ='nearest')

            data = np.transpose(data,[2,1,0])
            # if np.isnan(data).any():
            #     print('data is none')
            seg_label = np.transpose(seg_label,[2,1,0])
            # if np.isnan(seg_label).any():
            #     print('label is none')

        if transpose_side_slice:
           data = data.transpose(0,2,1)
           seg_label = seg_label.transpose(0,2,1)

        if self.subtract_mean:
            data -= 127.0


        if self.transform:
            data, seg_label = self.transform(data, seg_label)
            data  = RandomContrast(0.3,1.1)(data)

        #print('d ={}, s ={}'.format(data.shape, seg_label.shape))
        return data, seg_label

    def set_phase(self, phase):
        raise NotImplementedError("Must be implemented in subclass !")

    def load_data(self):
        '''Subclass must load data
         into 2 list of numpy array of dictionary in self.im_ld_pairs
         key = 'image' & 'label' '''
        raise NotImplementedError("Must be implemented in subclass !")

    @property
    def subset(self):
        return {'Set_A', 'Set_B', 'Set_C'}

    @property
    def name(self):
        return 'Dataset-CRIME-' + self.sub_dataset

    def __len__(self):
        self.len = self.x_size * self.y_size * self.z_size
        return self.len


class CRIME_Dataset(exp_Dataset):
    """ EM dataset."""

    # Initialize EM data
    def __init__(self,
                 in_patch_size = None,
                 out_patch_size=(224, 224, 1),
                 sub_dataset='Set_A',
                 subtract_mean=True,
                 phase='train',
                 channel_axis=2,
                 label_config = None,
                 transform=None,
                 data_config='conf/cremi_datasets_with_tflabels.toml'):

        self.data_config = data_config
        super(CRIME_Dataset, self).__init__(sub_dataset=sub_dataset,
                                            in_patch_size=in_patch_size,
                                            out_patch_size=out_patch_size,
                                            subtract_mean=subtract_mean,
                                            phase=phase,
                                            label_config =label_config,
                                            channel_axis =channel_axis,
                                            transform=transform)

    def __getitem__(self, index):
        im_data, lb_data = self.random_choice_dataset(self.im_lb_pair)
        data, seg_label = self.get_random_patch(index, im_data, lb_data)
        if self.channel_axis ==1:
            transpose_d =  [self.channel_axis,0,2]
        elif self.channel_axis ==2:
            transpose_d =  [self.channel_axis,0,1]
        
        if self.channel_axis >0:
            data=data.transpose(transpose_d)
            seg_label=seg_label.transpose(transpose_d)
        '''Convert seg_label to 2D by obtaining only intermedia slice 
           while the input data have multiple slice as multi-channel input
           the network only output the prediction of sigle slice in the center of Z dim'''
        if seg_label.ndim == 3:
            #z_dim = seg_label.shape[self.channel_axis]
            ch_dim = seg_label.shape[0]
            assert ((ch_dim % 2) == 1)  # we will ensure that # slices is odd number
            m_slice_idx = ch_dim // 2
            # seg_label      = seg_label[m_slice_idx,:,:]
            seg_label = np.expand_dims(seg_label[m_slice_idx, :, :], axis=0)
        tc_data = torch.from_numpy(data).float()
        tc_label_dict = self.label_generator(seg_label)[0]
        tc_label_dict['seg'] = seg_label
        return tc_data, tc_label_dict

    def set_phase(self, phase):
        self.phase = phase
        if phase == 'train':
            self.slice_start_z = 0
            self.slice_end_z = 99
        elif phase == 'valid':
            self.slice_start_z = 100
            self.slice_end_z = 124

        self.z_size = self.slice_end_z - self.slice_start_z + 1

    def load_data(self):

        #data_config = 'conf/cremi_datasets.toml'
        volumes = HDF5Volume.from_toml(self.data_config)
        if 'tflabels' in self.data_config:
            data_name = {'Set_A': 'Sample A with extra transformed labels',
                         'Set_B': 'Sample B with extra transformed labels',
                         'Set_C': 'Sample C with extra transformed labels'
                         }
        else:
            data_name = {'Set_A':'Sample A', 
                         'Set_B':'Sample B',
                         'Set_C':'Sample C'
                         }

        im_lb_pair = {}
        if self.sub_dataset == 'All':
            for k, v in data_name.iteritems():
                V = volumes[data_name[k]]
                im_lb_pair[k] = {'image': V.data_dict['image_dataset'],
                                 'label': V.data_dict['label_dataset']}
        else:
            k = self.sub_dataset
            V = volumes[data_name[k]]
            im_lb_pair[k] = {'image': V.data_dict['image_dataset'],
                             'label': V.data_dict['label_dataset']}

        return im_lb_pair


class CRIME_Dataset3D(exp_Dataset):
    """ EM dataset."""

    # Initialize EM data
    def __init__(self,
                 out_patch_size=(224, 224, 1),
                 sub_dataset='Set_A',
                 subtract_mean=True,
                 phase='train',
                 transform=None,
                 channel_axis=1,
                 label_config = None,
                 predict_patch_size=(320,320,25),
                 predict_overlap =(30,30, 8),
                 data_config='conf/cremi_datasets_with_tflabels.toml'):


        #pdb.set_trace()

        self.data_config = data_config
        super(CRIME_Dataset3D, self).__init__(sub_dataset=sub_dataset,
                                            out_patch_size=out_patch_size,
                                            subtract_mean=subtract_mean,
                                            phase=phase,
                                            transform=transform)
        self.predict_patch_size = predict_patch_size
        self.predict_overlap    = predict_overlap
        self.label_generator = label_transform3D(label_config)

    def __getitem__(self, index):
        
        def expand_dims_in_dict(in_dict):
            for k,v  in in_dict.iteritems():
                if v.ndim ==3:
                    in_dict[k] = torch.from_numpy(np.expand_dims(v,axis=0)).float()
                else:
                    in_dict[k] = torch.from_numpy(v).float()
            return in_dict
        
        im_data, lb_data = self.random_choice_dataset(self.im_lb_pair)
        data, seg_label = self.get_random_patch(index, im_data, lb_data)
        data = np.expand_dims(data, axis=0)
        tc_data = torch.from_numpy(data).float()
        tc_label_dict = self.label_generator(seg_label)[0]
        #tc_label_dict['seg'] = torch.from_numpy(seg_label).float()
        tc_label_dict['seg'] = seg_label.astype(np.float)
        tc_label_dict=expand_dims_in_dict(tc_label_dict)

        return tc_data, tc_label_dict


    # def _get_patch_data_(self,starting_point, size):
    #     z_start, y_start, x_start = \
    #                 start_pose[0],starting_point[1],starting_point[2]
    #     z_end, y_end, x_end = \
    #                 z_start + size[0], y_start+size[1], x_start+size[2]
    #     cur_set = self.im_lb_pair[self.current_subDataset]
    #     out= {}
    #     data = np.array(im_data[z_start:z_end, y_start:y_end, x_start:x_end]).astype(np.float)
    #     if self.subtract_mean:
    #         data -=127.0
    #     out['data']=data

    #     if 'label' in cur_set:
    #         seg_label = np.array(lb_data[z_start:z_end, y_start:y_end, x_start:x_end]).astype(np.int)
    #         out['label'] =seg_label

    #     return out


    def get_label(self):
        if not 'label' in self.im_lb_pair[self.current_subDataset]:
            return None
        else:
            return self.im_lb_pair[self.current_subDataset]['label']

    def set_current_subDataset(self,data_name):
        self.current_subDataset =data_name

    def conv_slice_3DPatch(self):

        def compute_slice(step,axis):
            return slice(step*stride[axis],step*stride[axis]+self.predict_patch_size[axis])
        import operator
        cur_set = self.im_lb_pair[self.current_subDataset]
        print(cur_set.keys())
        data_size = cur_set['image'].shape
        overlap_count_array = np.zeros(data_size).astype(np.float)
        n_dim =3
        assert len(data_size) == n_dim
        ''' the slice oder : x, y, z '''

        stride = list(map(operator.sub, self.predict_patch_size, self.predict_overlap))
        z_steps, y_steps, x_steps = [(d_size - p_size)/s_stride+ 1 for p_size, (d_size, s_stride) in zip(self.predict_patch_size,zip(data_size,stride))]

        residaul = [d_size%s_stride for d_size, s_stride in zip(data_size,stride)]

        #residual = r>0
        #pdb.set_trace()
        # z_steps = data_size[0] / stride[0]
        # y_steps = data_size[1] / stride[1]
        # x_steps = data_size[2] / stride[2]

        slice_list = []
        for z in range(z_steps+1):
            slice_3d = [slice(None)]*n_dim
            slice_z = compute_slice(z,axis=0)
            slice_3d[0]=slice_z
            for y in range(y_steps):
                slice_y = compute_slice(y,axis=1)
                slice_3d =list(slice_3d)
                slice_3d[1]=slice_y
                for x in range(x_steps):
                    slice_x = compute_slice(x,axis=2)
                    slice_3d=list(slice_3d) # make a copy
                    slice_3d[2]=slice_x
                    slice_list.append(slice_3d)
                    #pdb.set_trace()
                    overlap_count_array[slice_3d]+=1

        #pdb.set_trace()
        out = {}
        im_data=np.array(cur_set['image'])

        if 'test' in self.data_config:
            print('replace bad slice in test set {}'.format(self.current_subDataset))
            im_data=replace_bad_slice_in_test(im_data,self.current_subDataset)



        #pdb.set_trace()
        out['image']=list(map(lambda x: im_data[x]-127.0, slice_list))
        if 'label' in cur_set:
            lb_data=np.array(cur_set['label'])
            out['label']=list(map(lambda x: lb_data[x], slice_list))

        return out, slice_list, overlap_count_array

    def assemble_conv_slice(self, patch_data_list, slices_list,overlap_count_array):
        cur_set = self.im_lb_pair[self.current_subDataset]

        
        #from matplotlib import pyplot as plt
        #pdb.set_trace()
        predict_channels =patch_data_list[0].shape[1]
        pred_size = [predict_channels] + list(cur_set['image'].shape)
        assembled_data = np.zeros(pred_size)
        for idx in range(len(slices_list)):
            #print('input slice = {}, input data shape ={}'.format(slices_list[idx],patch_data_list[idx].shape))
            new_slice=[slice(None)]+slices_list[idx]
            assembled_data[new_slice]+= np.squeeze(patch_data_list[idx])
            #print('path_data {} = {}'.format(idx,patch_data_list[idx].shape))




        assembled_data =  assembled_data / overlap_count_array

        return assembled_data










    def set_phase(self, phase):
        self.phase = phase
        if phase == 'train':
            self.slice_start_z = 0
            self.slice_end_z = 99
        elif phase == 'valid':
            self.slice_start_z = 100
            self.slice_end_z = 124

        self.z_size = self.slice_end_z - self.slice_start_z + 1

    def load_data(self):

        #self.data_config = 'conf/cremi_datasets.toml'
        volumes = HDF5Volume.from_toml(self.data_config)
        # data_name = {'Set_A': 'Sample A with extra transformed labels',
        #              'Set_B': 'Sample B with extra transformed labels',
        #              'Set_C': 'Sample C with extra transformed labels'
        #              }
        # # data_name = {'Set_A':'Sample A',
        # #              'Set_B':'Sample B',
        # #              'Set_C':'Sample C'
        # #             }

        if 'tflabels' in self.data_config:
            data_name = {'Set_A': 'Sample A with extra transformed labels',
                         'Set_B': 'Sample B with extra transformed labels',
                         'Set_C': 'Sample C with extra transformed labels'
                         }
        else:
            data_name = {'Set_A':'Sample A', 
                         'Set_B':'Sample B',
                         'Set_C':'Sample C'
                         }

        volume_names = data_name if self.sub_dataset == 'All' \
                       else {self.dataset:data_name[self.dataset]}

        im_lb_pair={}
        for key,vname in volume_names.iteritems():
            V = volumes[vname]
            if 'label_dataset' in V.data_dict:
                im_lb_pair[key] = {'image': V.data_dict['image_dataset'],
               'label': V.data_dict['label_dataset']}
            else:
               im_lb_pair[key] = {'image': V.data_dict['image_dataset']}
        return im_lb_pair



        # im_lb_pair = {}
        # if self.sub_dataset == 'All':
        #     for k, v in data_name.iteritems():
        #         V = volumes[data_name[k]]
        #         im_lb_pair[k] = {'image': V.data_dict['image_dataset'],
        #                          'label': V.data_dict['label_dataset']}
        # else:
        #     k = self.sub_dataset
        #     V = volumes[data_name[k]]
        #     im_lb_pair[k] = {'image': V.data_dict['image_dataset'],
        #                      'label': V.data_dict['label_dataset']}

        # return im_lb_pair


class CRIME_slice_merge_dataset(CRIME_Dataset):
    def __init__(self,
                 out_patch_size=(112, 112, 2),
                 in_patch_size =(640,640,2),
                 sub_dataset='Set_A',
                 subtract_mean=True,
                 phase='train',
                 transform=None,
                 data_config='conf/cremi_datasets_with_tflabels.toml'):
        super(CRIME_Dataset_3D_labels, self).__init__(sub_dataset=sub_dataset,
                                                      out_patch_size=out_patch_size,
                                                      in_patch_size =in_patch_size,
                                                      subtract_mean=subtract_mean,
                                                      phase=phase,
                                                      transform=transform,
                                                      data_config=data_config)

    def __getitem__(self, index):
        im_data, lb_data = self.random_choice_dataset(self.im_lb_pair)
        data, seg_label  = self.get_random_patch(index, im_data, lb_data)
        tc_data = torch.from_numpy(data).float()
        tc_label_dict = self.gen_label_per_slice(seg_label)
        # for k,v in tc_label_dict.iteritems():
        #     print( '{} shape = {}'.format(k,v.shape))
        return tc_data, seg_label, tc_label_dict

    def gen_label_per_slice(self, seg):
        if seg.ndim == 3:
            z_dim = seg.shape[0]
            assert ((z_dim % 2) == 1)  # need ensure that # slices is odd number
            slice_seg_list = [seg[i, :, :] for i in range(z_dim)]
            slice_tgDict_list = self.label_generator(*slice_seg_list)
            lb_dict = {}
            for k in slice_tgDict_list[0].keys():
                data_list  =[]
                for i in range(len(slice_seg_list)):
                    d  = slice_tgDict_list[i][k]
                    if d.dim() ==2:
                        d =torch.unsqueeze(d,0)
                    assert(d.dim() ==3)
                    data_list.append(d)
                lb_dict[k] = torch.cat(data_list,dim =0)
                #lb_dict[k] = torch.cat([slice_tgDict_list[i][k] for i in range(len(slice_seg_list))], dim=0)
            return lb_dict




class slice_dataset(Dataset):
    def __init__(self, sub_dataset='Set_A',
                 subtract_mean=True,
                 split='valid',
                 slices = 3,
                 slice_axis = 0,
                 data_config='conf/cremi_datasets.toml'):
                 #data_config='conf/cremi_datasets_with_tflabels.toml'):
        self.sub_dataset = sub_dataset
        self.slices = slices
        self.subtract_mean = subtract_mean
        self.data_config   = data_config
        self.slice_axis = slice_axis
        #self.starting_slice = 100 if split == 'valid' else 0
        #self.starting_slice = 61 if split == 'valid' else 0
        self.starting_slice = 0
        self.im_lb_pair = self.load_data()
        self.im_lb_pair ={k : {k2: self.transpose_source_data(v) for k2,v in k_dict.iteritems()}
                          for k, k_dict in self.im_lb_pair.iteritems()}

        im_data = self.im_lb_pair[self.im_lb_pair.keys()[0]]['image']

        self.data_shape = im_data.shape
        self.y_size = self.data_shape[2]
        self.x_size = self.data_shape[1]
        self.z_size = self.data_shape[0]

        if sub_dataset != 'All':
            self.set_current_subDataset(sub_dataset)

    def transpose_source_data(self, val_3d):
        if self.slice_axis == 0:
            return val_3d
        elif self.slice_axis == 1:
            transpose_idx = [1,0,2]
        elif self.slice_axis == 2:
            transpose_idx = [2,0,1]
        else:
          raise  ValueError('slice_axis is out of range {}'.format(slice_axis))
        return np.transpose(val_3d, transpose_idx)
    
    def reserve_transpose_data(self,val_3d):
        if self.slice_axis == 0:
            return val_3d
        elif self.slice_axis ==1:
            transpose_idx = [1,0,2]
        elif self.slice_axis ==2:
            transpose_idx = [1,2,0]
        else:
          raise  ValueError('slice_axis is out of range {}'.format(slice_axis))
        return np.transpose(val_3d, transpose_idx)


    def set_current_subDataset(self,data_name):
        self.current_subDataset =data_name
    def set_slice_direction(self,direction):
        self.slice_direction =direction

    @property
    def subset(self):
        return self.im_lb_pair.keys()

    def output_labels(self):
        ''' output: diction, Key = name of label, value = channel of output '''
        return {'gradient': 2, 'affinity': 1, 'centermap': 2, 'sizemap': 1, 'distance': 1,'skeleton':1}

    def get_data(self):
        return self.im_lb_pair[self.current_subDataset]['image'][self.starting_slice:,:,:]
    
    def get_label(self):
        if not 'label' in self.im_lb_pair[self.current_subDataset]:
            return None
        else:
            return self.im_lb_pair[self.current_subDataset]['label'][self.starting_slice:,:,:]

    def __get_side_slices__(self,index, data_scr='image',axis=1):
        cur_set = self.im_lb_pair[self.current_subDataset]
        im_data = cur_set[data_scr]
        start_idx = index - self.slices // 2
        end_idx  = start_idx + self.slices
        print('get slice {} start ={} end ={}'.format(index, start_idx, end_idx))
        def get_side_data(input_array,start_idx,end_idx):
            if start_idx < 0:
                start_idx = 0
                if axis ==1:
                    r_slice = input_array[self.starting_slice:, start_idx:start_idx+1,:]
                    d_slice = input_array[self.starting_slice:, start_idx:end_idx,:]
                elif axis ==2:
                    r_slice = input_array[self.starting_slice:, :,start_idx:start_idx+1]
                    d_slice = input_array[self.starting_slice:, :,start_idx:end_idx]
                for i in range(self.slices - (end_idx-start_idx)):
                  d_slice  =  np.concatenate([r_slice,d_slice],axis)
                #d_slice = np.transpose(d_slice,[1,0,2])
                #print('after d shape = {}'.format(d_slice.shape))
            elif end_idx >im_data.shape[axis]:
                end_idx  = im_data.shape[axis]
                if axis ==1:
                    r_slice = input_array[self.starting_slice:, -1:,:]
                    d_slice = input_array[self.starting_slice:, start_idx:,:]
                elif axis ==2:
                    r_slice = input_array[self.starting_slice:, :,-1:]
                    d_slice = input_array[self.starting_slice:, :,start_idx:]
                for i in range(self.slices - (end_idx-start_idx)):
                  d_slice  =  np.concatenate([d_slice,r_slice],axis)
                #d_slice = np.transpose(d_slice,[2,0,1])
                #print('after d shape = {}'.format(d_slice.shape))
            else:
                if axis ==1:
                    d_slice = input_array[self.starting_slice:,start_idx:end_idx,:]
                elif axis ==2:
                    d_slice = input_array[self.starting_slice:,:,start_idx:end_idx]
            
            transpose_seq = [1,0,2] if axis == 1 else [2,0,1]
            d_slice = np.transpose(d_slice, transpose_seq)
            return d_slice
        
        im_slice=get_side_data(im_data,start_idx,end_idx)
        return im_slice
    
    def __get_front_slices__(self,index,data_scr='image'):
        cur_set = self.im_lb_pair[self.current_subDataset]
        im_data = cur_set[data_scr]
        start_idx = index - self.slices // 2 + self.starting_slice
        end_idx  = start_idx + self.slices
        def get_slice_data(input_array,start_idx,end_idx):
            if start_idx < self.starting_slice:
                start_idx = self.starting_slice
                r_slice = input_array[start_idx:start_idx+1, :,:]
                d_slice = input_array[start_idx:end_idx,:,:]
                for i in range(self.slices - (end_idx-start_idx)):
                  d_slice  =  np.concatenate([r_slice,d_slice],0)
                #print('after d shape = {}'.format(d_slice.shape))
            elif end_idx > self.z_size:
                end_idx = self.z_size
                r_slice = input_array[-1:, :,:]
                d_slice = input_array[start_idx:,:,:]
                #print('d shape = {}'.format(d_slice.shape))
                #print('r shape = {}'.format(r_slice.shape))
                #print('start idx = {}, end_idx ={} z_size= {}'.format(start_idx,end_idx,self.z_size))
                for i in range(self.slices - (self.z_size-start_idx)):
                  d_slice  =  np.concatenate([d_slice,r_slice],0)
            else:
                d_slice = input_array[start_idx:end_idx,:,:]
            return d_slice
        im_slice=get_slice_data(im_data,start_idx,end_idx)
        return im_slice



    def __getitem__(self, index):
        output ={}
        cur_set = self.im_lb_pair[self.current_subDataset]
        im_slice=self.__get_front_slices__(index, data_scr = 'image')
        im_data = cur_set['image']
        #im_slice = self.__get_front_slices__(index)
        im_slice = im_slice.astype(np.float)
        if self.subtract_mean:
            im_slice -= 127.0
        im_slice = np.expand_dims(im_slice,axis=0)
        output['data'] = torch.from_numpy(im_slice).float()
        if 'label' in cur_set:
            lb_data  = cur_set['label']
            lb_slice=self.__get_front_slices__(index,data_scr='label')
            lb_slice = np.expand_dims(lb_slice,axis=0)
            lb_slice = lb_slice.astype(np.int)
            output['label'] = torch.from_numpy(lb_slice).float()
        return output


        # output ={}
        # cur_set = self.im_lb_pair[self.current_subDataset]
        # im_slice=get_slice_data(im_data,start_idx,end_idx)
        # im_data = cur_set['image']
        # if direction == 'high_res':
        #     im_slice = self.__get_front_slices__(index)
        #     print('getting front slice')
        # else:
        #     print('getting side slice')
        #     im_slice=self.__get_side_slices__(index)
        # im_slice = im_slice.astype(np.float)
        # if self.subtract_mean:
        #     im_slice -= 127.0
        # im_slice = np.expand_dims(im_slice,axis=0)
        # output['data'] = torch.from_numpy(im_slice).float()
        # #print('output[data] = {}'.format(output['data'].shape))
        # # plt.imshow(np.squeeze(im_slice[0,1]))
        # # plt.show()

        # if 'label' in cur_set:
        #     lb_data  = cur_set['label']
        #     if direction == 'high_res':
        #         lb_slice=self.__get_front_slices__(index,data_scr='label')
        #     else:
        #         lb_slice=self.__get_side_slices__(index,data_scr='label')
        #     lb_slice = np.expand_dims(lb_slice,axis=0)
        #     lb_slice = lb_slice.astype(np.int)
        #     output['label'] = torch.from_numpy(lb_slice).float()
        # return output

    def __len__(self):
        return self.z_size
        # if self.slice_direction == 'high_res':
        #     return self.z_size - self.starting_slice
        # else:
        #     return self.x_size


    def load_data(self):

        # data_config = 'conf/cremi_datasets.toml'
        volumes = HDF5Volume.from_toml(self.data_config)
        if 'tflabels' in self.data_config:
            data_name = {'Set_A': 'Sample A with extra transformed labels',
                         'Set_B': 'Sample B with extra transformed labels',
                         'Set_C': 'Sample C with extra transformed labels'
                         }
        else:
            data_name = {'Set_A':'Sample A',
                         'Set_B':'Sample B',
                         'Set_C':'Sample C'
                         }

        im_lb_pair = {}
        print('volume = {}'.format(volumes.keys()))
        
        volume_names = data_name if self.sub_dataset == 'All' \
                       else {self.sub_dataset:data_name[self.sub_dataset]}

        for key,vname in volume_names.iteritems():
            V = volumes[vname]
            if 'label_dataset' in V.data_dict:
                im_lb_pair[key] = {'image': V.data_dict['image_dataset'],
               'label': V.data_dict['label_dataset']}
            else:
               im_lb_pair[key] = {'image': V.data_dict['image_dataset']}
        return im_lb_pair

        


        # if self.sub_dataset == 'All':
        #     for k, ~ in data_name.iteritems():
        #         V = volumes[data_name[k]]
        #         if 'label_dataset' in V.data_dict:
        #             im_lb_pair[k] = {'image': V.data_dict['image_dataset'],
        #             'label': V.data_dict['label_dataset']}
        #         else:
        #              im_lb_pair[k] = {'image': V.data_dict['image_dataset']}

        # else:
        #     k = self.sub_dataset
        #     V = volumes[data_name[k]]
        #     if 'label_dataset' in V.data_dict:
        #         im_lb_pair[k] = {'image': V.data_dict['image_dataset'],
        #                          'label': V.data_dict['label_dataset']}
        #     else:
        #         im_lb_pair[k] = {'image': V.data_dict['image_dataset']}

        
class labelGenerator(object):
    def __init__(self,label_config):
        self.label_generator = label_transform(label_config=label_config)

    def __call__(self, *input):
        ''' set distance large enough to conver the boundary
         as to put more weight on bouday areas.
         which is important when do cut for segmentation '''
        ''' compute the runtime obj graidient instead of pre-computed one
        to avoid using wrong gradient map when performing data augmentation
        such as flip, rotate etc.'''

        """ Input: segmentation label """

        """ Ouput: list of transformed labels dict:  
                     d{'gradient','affinity','centermap','sizemap','distance'}):
                     """

        output = []
        for idex, seg_label in enumerate(input):
            # print ('seg_label shape = {}'.format(seg_label.shape))
            affinMap = self.affinityFunc(seg_label)
            objCenterMap = self.objCenterFunc(seg_label)

            trans_data_list = self.label_generator(seg_label)
            grad_x, grad_y = trans_data_list[0]['gradient']
            grad = np.concatenate((grad_x, grad_y), 0)

            tc_label_dict = {}
            for key, value in trans_data_list[0].iteritems():
                tc_label_dict[key] = torch.from_numpy(value).float() \
                    if key is not 'gradient' \
                    else torch.from_numpy(grad).float()
            tc_label_dict['affinity'] = torch.from_numpy(affinMap).float()
            tc_label_dict['centermap'] = torch.from_numpy(objCenterMap).float()

            output.append(tc_label_dict)
        return output

    def output_labels(self):
        # type: () -> object
        ''' output: diction, Key = name of label, value = channel of output '''
        return {'gradient': 2, 'affinity': 1, 'centermap': 2, 'sizemap': 1, 'distance': 1, 'skeleton': 1}

    def affinityFunc(self, seg_label):
        affineX = affinity(axis=-1, distance=2)
        affineY = affinity(axis=-2, distance=2)
        affinMap = ((affineX(seg_label)[0] + affineY(seg_label)[0]) > 0).astype(np.int)
        return affinMap

    def objCenterFunc(self, seg_label):
        centermap = objCenterMap()
        [(x_centerMap, y_centerMap)] = centermap(seg_label)
        objCenter_map = np.concatenate((x_centerMap, y_centerMap), 0)
        return objCenter_map


class labelGenerator3D(object):
    def __init__(self, label_config=None):
        self.label_generator = label_transform3D(label_config =label_config)

    def __call__(self, *input):
        ''' set distance large enough to conver the boundary
         as to put more weight on bouday areas.
         which is important when do cut for segmentation '''
        ''' compute the runtime obj graidient instead of pre-computed one
        to avoid using wrong gradient map when performing data augmentation
        such as flip, rotate etc.'''

        """ Input: segmentation label """

        """ Ouput: list of transformed labels dict:  
                     d{'gradient','affinity','centermap','sizemap','distance'}):
                     """

        output = []
        for idex, seg_label in enumerate(input):
            # print ('seg_label shape = {}'.format(seg_label.shape))
            trans_data_list = self.label_generator(seg_label)
            tc_label_dict = {}
            # for 3D only 'distance key'
            for key, value in trans_data_list[0].iteritems():
                tc_label_dict[key] = torch.from_numpy(value).float()
           
            output.append(tc_label_dict)
        return output

    def output_labels(self):
        return self.label_generator.output_labels()
        # type: () -> object
        ''' output: diction, Key = name of label, value = channel of output '''
        #return {'distance': 1}



def saveGradfiguers(iters, file_prefix, output):
    my_dpi = 96
    plt.figure(figsize=(1250 / my_dpi, 1250 / my_dpi), dpi=my_dpi)
    # print ('tc data output shape = {}'.format(output.shape))
    data = output.numpy()
    # print ('output shape = {}'.format(data.shape))
    I = data[0, 0, :, :]
    plt.imshow(I)
    plt.savefig(file_prefix + '_readerX{}.png'.format(iters))
    I = data[0, 1, :, :]
    plt.imshow(I)
    plt.savefig(file_prefix + '_readerY{}.png'.format(iters))


def saveRawfiguers(iters, file_prefix, output):
    my_dpi = 96
    plt.figure(figsize=(1250 / my_dpi, 1250 / my_dpi), dpi=my_dpi)
    data = output.numpy()
    print ('output shape = {}'.format(data.shape))
    I = data[0, :, :]
    plt.imshow(I)
    # pdb.set_trace()
    plt.savefig(file_prefix + '_raw{}.png'.format(iters))


def l2_norm(x):
    # epsilon=torch.cuda.DoubleTensor([1e-12])
    # sq_x   = torch.max(x**2,epsilon)
    # sq_x   = torch.max(x**2,epsilon)
    # e_mat  = torch.zero_like(sq_x)
    sum_x = torch.sum(x ** 2, 1, keepdim=True)
    sqrt_x = torch.sqrt(sum_x)
    return x / sqrt_x


def compute_angular(x):
    x = F.normalize(x) * 0.99999
    # print('x shpe in angu {}'.format(x.shape))
    x_aix = x[:, 0, :, :] / torch.sqrt(torch.sum(x ** 2, 1))
    # print('x aix shape {}'.format(x_aix.shape))
    angle_map = torch.acos(x_aix)
    # print('angle_map shape {}'.format(angle_map.shape))
    return angle_map


def test_angluar_map():
    data_config = '../conf/cremi_datasets_with_tflabels.toml'
    dataset = CRIME_Dataset(data_config=data_config, phase='valid')
    train_loader = DataLoader(dataset=dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1)
    for i, data in enumerate(train_loader, start=0):
        inputs, labels = data
        labels = labels[:, 0, :, :, :]
        ang_map = compute_angular(labels)
        saveRawfiguers(i, 'ang_map_test', ang_map)
        if i > 3:
            break


def test_transform():
    data_config = '../conf/cremi_datasets_with_tflabels.toml'
    # data_config = '../conf/cremi_datasets.toml'
    trans = random_transform(VFlip(), HFlip(), Rot90())
    dataset = CRIME_Dataset(data_config=data_config, phase='valid', transform=trans)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1)
    for i, (inputs, target) in enumerate(train_loader, start=0):
        # for i , all in enumerate(train_loader,start =0):

        # labels = labels[:,0,:,:,:]
        # print inputs.shape
        im = inputs[0, 0].numpy()
        # pdb.set_trace()
        # tg1 = target['gradient'][0,0].numpy()
        # tg2 = target['gradient'][0,1].numpy()
        ang_map = compute_angular(target['gradient'])[0].numpy()

        for key, value in target.iteritems():
            print('{} shape is {}'.format(key, value.shape))

        # print('ang_mp shape = {}'.format(ang_map.shape))
        # print('dist shape {}'.format(target['distance'].shape))
        dist = target['distance'][0].numpy()
        sizemap = target['sizemap'][0].numpy()
        affinity = target['affinity'][0].numpy()
        centermap = target['centermap'][0].numpy()
        dist = np.squeeze(dist)
        sizemap = np.squeeze(sizemap)
        affinity = np.squeeze(affinity)
        center_x = np.squeeze(centermap[0])
        center_y = np.squeeze(centermap[1])
        # print('center shape ={}'.format(center.shape))

        fig, axes = plt.subplots(nrows=2, ncols=3, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
        axes[0, 0].imshow(im, cmap='gray')
        axes[0, 0].axis('off')
        axes[0, 0].margins(0, 0)

        axes[0, 1].imshow(affinity)
        axes[0, 1].axis('off')
        axes[0, 1].margins(0, 0)

        axes[1, 0].imshow(dist)
        axes[1, 0].axis('off')
        axes[1, 0].margins(0, 0)

        axes[1, 1].imshow(np.log(sizemap))
        axes[1, 1].axis('off')
        axes[1, 1].margins(0, 0)

        axes[0, 2].imshow(center_x)
        axes[0, 2].axis('off')
        axes[0, 2].margins(0, 0)

        axes[1, 2].imshow(center_y)
        axes[1, 2].axis('off')
        axes[1, 2].margins(0, 0)

        plt.margins(x=0.001, y=0.001)
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.show()
        # plt.close('all')
        if i > 5:
            break


if __name__ == '__main__':
    test_transform()
