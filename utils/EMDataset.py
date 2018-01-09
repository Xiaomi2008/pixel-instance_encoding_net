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
import pdb


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
                 label_gen=None):

        self.sub_dataset = sub_dataset
        self.phase = phase
        self.channel_axis = (channel_axis + 1) % 3

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
        self.y_size = dim_shape[2] - self.x_out_size + 1
        self.x_size = dim_shape[1] - self.y_out_size + 1

        # self.label_generator  = label_transform(objSizeMap =True)
        self.label_generator = label_gen if label_gen else labelGenerator()

    def output_labels(self):
        return self.label_generator.output_labels()
        # return ['gradient','affinity','centermap','sizemap','distance']

    def __getitem__(self, index):

        # random choice one of sub_datasets
        im_data, lb_data = self.random_choice_dataset(self.im_lb_pair)
        data, seg_label = self.get_random_patch(index, im_data, lb_data)

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

        im_z_size, im_x_size, im_y_size = im_data.shape[0], im_data.shape[1], im_data.shape[2]
        if im_z_size - self.slice_start_z -1 < self.z_out_size:
            z_start = self.slice_start_z
            z_end   = im_z_size
        else:
            z_start = np.random.randint(self.slice_start_z, self.slice_end_z - self.z_out_size -1)
            z_end   = z_start +  self.z_out_size
       
        x_start = np.random.randint(im_x_size - self.x_in_size - 1)
        y_start = np.random.randint(im_y_size - self.y_in_size - 1)


        if z_start > im_z_size - self.z_out_size:
            z_start = im_z_size - self.z_out_size

       
        x_end = x_start + self.x_out_size
        y_end = y_start + self.y_out_size

        # assert (z_end - z_start == 3)
        #print ('y_e={}, y_s ={}'.format(y_start, y_end))
        if self.channel_axis >0:
            idx=np.random.randint(2)
            if idx == 0:
                data = np.array(im_data[z_start:z_end, x_start:x_end, y_start:y_end]).astype(np.float)
                seg_label = np.array(lb_data[z_start:z_end, x_start:x_end, y_start:y_end]).astype(np.int)
            else:
                data = np.array(im_data[x_start:x_end, z_start:z_end, y_start:y_end]).astype(np.float)
                seg_label = np.array(lb_data[x_start:x_end, z_start:z_end, y_start:y_end]).astype(np.int)
                data = data.transpose(1,0,2)
                seg_label = seg_label.transpose(1,0,2)

        if self.subtract_mean:
            data -= 127.0


        if self.transform:
            data, seg_label = self.transform(data, seg_label)
        # print data.shape
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
                 out_patch_size=(224, 224, 1),
                 sub_dataset='Set_A',
                 subtract_mean=True,
                 phase='train',
                 channel_axis = 2,
                 transform=None,
                 data_config='conf/cremi_datasets_with_tflabels.toml'):

        self.data_config = data_config
        super(CRIME_Dataset, self).__init__(sub_dataset=sub_dataset,
                                            out_patch_size=out_patch_size,
                                            subtract_mean=subtract_mean,
                                            phase=phase,
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

        # data_config = 'conf/cremi_datasets.toml'
        volumes = HDF5Volume.from_toml(self.data_config)
        data_name = {'Set_A': 'Sample A with extra transformed labels',
                     'Set_B': 'Sample B with extra transformed labels',
                     'Set_C': 'Sample C with extra transformed labels'
                     }
        # data_name = {'Set_A':'Sample A',
        #              'Set_B':'Sample B',
        #              'Set_C':'Sample C'
        #             }
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


class slice_dataset(Dataset):
    def __init__(self, sub_dataset='Set_A',
                 subtract_mean=True,
                 split='valid',
                 slices = 3,
                 data_config='conf/cremi_datasets.toml'):
                 #data_config='conf/cremi_datasets_with_tflabels.toml'):
        self.sub_dataset = sub_dataset
        self.slices = slices
        self.subtract_mean = subtract_mean
        self.data_config   = data_config
        self.starting_slice = 100 if split == 'valid' else 0
        self.im_lb_pair = self.load_data()
        im_data = self.im_lb_pair[self.im_lb_pair.keys()[0]]['image']
        self.data_shape = im_data.shape
        self.y_size = self.data_shape[2]
        self.x_size = self.data_shape[1]
        self.z_size = self.data_shape[0]

    def set_current_subDataset(self,data_name):
        self.current_subDataset =data_name

    @property
    def subset(self):
        return self.im_lb_pair.keys()

    def output_labels(self):
        ''' output: diction, Key = name of label, value = channel of output '''
        return {'gradient': 2, 'affinity': 1, 'centermap': 2, 'sizemap': 1, 'distance': 1}

    def get_data(self):
        return self.im_lb_pair[self.current_subDataset]['image'][self.starting_slice:,:,:]
    def get_label(self):
        if not 'label' in self.im_lb_pair[self.current_subDataset]:
            return None
        else:
            return self.im_lb_pair[self.current_subDataset]['label'][self.starting_slice:,:,:]


    def __getitem__(self, index):
        cur_set = self.im_lb_pair[self.current_subDataset]
        im_data = cur_set['image']
        start_idx = index - self.slices // 2 + self.starting_slice
        end_idx  = start_idx + self.slices
        def get_slice_data(input_array,start_idx,end_idx):
            if start_idx < self.starting_slice:
                start_idx = self.starting_slice
                r_slice = input_array[start_idx:start_idx+1, :,:]
                d_slice = input_array[start_idx:end_idx,:,:]
                for i in range(self.slices - (end_idx-start_idx)):
                  d_slice  =  np.concatenate([r_slice,d_slice],0)
                print('after d shape = {}'.format(d_slice.shape))
            elif end_idx > self.z_size:
                end_idx = self.z_size
                r_slice = input_array[-1:, :,:]
                d_slice = input_array[start_idx:,:,:]
                print('d shape = {}'.format(d_slice.shape))
                print('r shape = {}'.format(r_slice.shape))
                print('start idx = {}, end_idx ={} z_size= {}'.format(start_idx,end_idx,self.z_size))
                for i in range(self.slices - (self.z_size-start_idx)):
                  d_slice  =  np.concatenate([d_slice,r_slice],0)
            else:
                d_slice = input_array[start_idx:end_idx,:,:]
            return d_slice

        output ={}
        im_slice=get_slice_data(im_data,start_idx,end_idx)
        im_slice=im_slice.astype(np.float)
        if self.subtract_mean:
            im_slice -= 127.0
        im_slice = np.expand_dims(im_slice,axis=0)
        output['data'] = torch.from_numpy(im_slice).float()
        # plt.imshow(np.squeeze(im_slice[0,1]))
        # plt.show()

        if 'label' in cur_set:
            lb_data  = cur_set['label']
            lb_slice = get_slice_data(lb_data,start_idx,end_idx)
            lb_slice = np.expand_dims(lb_slice,axis=0)
            lb_slice = lb_slice.astype(np.int)
            output['label'] = torch.from_numpy(lb_slice).float()
        return output

    def __len__(self):
        return self.z_size - self.starting_slice


    def load_data(self):

        # data_config = 'conf/cremi_datasets.toml'
        volumes = HDF5Volume.from_toml(self.data_config)
        # data_name = {'Set_A': 'Sample A with extra transformed labels',
        #              'Set_B': 'Sample B with extra transformed labels',
        #              'Set_C': 'Sample C with extra transformed labels'
        #              }
        data_name = {'Set_A':'Sample A',
                     'Set_B':'Sample B',
                     'Set_C':'Sample C'
                     }
        im_lb_pair = {}
        print('volume = {}'.format(volumes.keys()))
        if self.sub_dataset == 'All':
            for k, v in data_name.iteritems():
                V = volumes[data_name[k]]
                if 'label_dataset' in V.data_dict:
                    im_lb_pair[k] = {'image': V.data_dict['image_dataset'],
                    'label': V.data_dict['label_dataset']}
                else:
                     im_lb_pair[k] = {'image': V.data_dict['image_dataset']}

        else:
            k = self.sub_dataset
            V = volumes[data_name[k]]
            #im_lb_pair[k] = {'image': V.data_dict['image_dataset'],
            #                 'label': V.data_dict['label_dataset']}
            if 'label_dataset' in V.data_dict:
                im_lb_pair[k] = {'image': V.data_dict['image_dataset'],
                                 'label': V.data_dict['label_dataset']}
            else:
                im_lb_pair[k] = {'image': V.data_dict['image_dataset']}

        return im_lb_pair

class labelGenerator(object):
    def __init__(self):
        self.label_generator = label_transform(objSizeMap=True)

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
        return {'gradient': 2, 'affinity': 1, 'centermap': 2, 'sizemap': 1, 'distance': 1}

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
