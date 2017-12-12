import os, sys
sys.path.append('../')
import pdb
import torch
import numpy as np
from transform import *
#import cv2
import torch.nn.functional as F
from matplotlib import pyplot as plt
#import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader
from label_transform.volumes import Volume
from label_transform.volumes import HDF5Volume
from label_transform.volumes import bounds_generator
from label_transform.volumes import SubvolumeGenerator

class exp_Dataset(Dataset):
    """base dataset"""
    def __init__(self, 
                out_patch_size = (3,224,224),
                in_patch_size  = None,
                sub_dataset    = 'All',
                subtract_mean  = True,
                phase          = 'train',
                transform      = None,
                label_gen      = None):
      

      self.sub_dataset      = sub_dataset
      self.phase            = phase

      self.x_out_size       = out_patch_size[1]
      self.y_out_size       = out_patch_size[2]
      self.z_out_size       = out_patch_size[0]
      self.in_patch_size    = in_patch_size if in_patch_size else out_patch_size
      self.x_in_size        = self.in_patch_size[0]
      self.y_in_size        = self.in_patch_size[1]
      self.z_in_size        = self.z_out_size

      self.subtract_mean    = subtract_mean
      self.transform        = transform
      
      ''' subclass should assign this param befor calling  __getitm__ function '''
      self.slice_start_z    = 0

      self.set_phase(phase)
      self.im_lb_pair= self.load_data()

      im_data  = self.im_lb_pair[self.im_lb_pair.keys()[0]]['image']
      dim_shape             = im_data.shape
      self.y_size           = dim_shape[2] -self.x_out_size + 1
      self.x_size           = dim_shape[1] -self.y_out_size + 1 

      #self.label_generator  = label_transform(objSizeMap =True)
      self.label_generator = label_generator if label_gen else  labelGenerator()

    def output_labels(self):
      return self.label_generator.output_labels()
      #return ['gradient','affinity','centermap','sizemap','distance']

    def __getitem__(self, index):

       # random choice one of sub_datasets
      im_data,lb_data=self.random_choice_dataset(self.im_lb_pair)
      data,seg_label =self.get_random_patch(index,im_data,lb_data)
      data = np.expand_dims(data, axis = 0)
      # seg_label = np.expand_dims(seg_label, axis = 0)
      # print('data shape is {}').format(data.shape)
      # print('label shape is {}').format(seg_label.shape)


      tc_data        = torch.from_numpy(data).float()
      tc_label_dict  = self.label_generator(seg_label)[0]
      return tc_data, tc_label_dict

    def random_choice_dataset(self,im_lb_pair):
      k=np.random.choice(im_lb_pair.keys())
      im_data = self.im_lb_pair[k]['image']
      lb_data = self.im_lb_pair[k]['label']  
      return im_data, lb_data
    
    def get_random_patch(self,index,im_data,lb_data):
      z_start = index // (self.x_size * self.y_size) + self.slice_start_z
      remain  = index % (self.x_size * self.y_size)
      x_start = remain // self.y_size
      y_start = remain % self.y_size

      z_end   = z_start + self.z_out_size
      x_end   = x_start + self.x_out_size
      y_end   = y_start + self.y_out_size

      data    = np.array(im_data[z_start:z_end,x_start:x_end,y_start:y_end]).astype(np.float)
      if self.subtract_mean:
        data -= 127.0
      
      seg_label   =np.array(lb_data[z_start:z_end,x_start:x_end,y_start:y_end]).astype(np.int)


      if self.transform:
        data,seg_label= self.transform(data,seg_label)

      return data, seg_label

    def set_phase(self,phase):
      raise NotImplementedError("Must be implemented in subclass !")

    def load_data(self):
      '''Subclass must load data 
         into 2 list of numpy array of dictionary in self.im_ld_pairs
         key = 'image' & 'label' '''
      raise NotImplementedError("Must be implemented in subclass !")

    @property
    def subset(self):
      return {'Set_A','Set_B','Set_C'}
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
                 out_patch_size       =   (1,224,224), 
                 sub_dataset          =   'Set_A',
                 subtract_mean        =   True,
                 phase                =   'train',
                 transform            =   None,
                 data_config          =   'conf/cremi_datasets_with_tflabels.toml'):
      
      self.data_config = data_config
      super(CRIME_Dataset,self).__init__(sub_dataset=sub_dataset, 
                                         out_patch_size =out_patch_size,
                                         subtract_mean =subtract_mean,
                                         phase = phase,
                                         transform =transform)
    
    def set_phase(self,phase):
      self.phase = phase
      if phase == 'train':
        self.slice_start_z= 0
        self.slice_end_z   = 99
      elif phase == 'valid':
        self.slice_start_z = 100
        self.slice_end_z = 124

      self.z_size = (self.slice_end_z - self.slice_start_z +1) - self.z_out_size + 1


        
    # def __getitem__(self, index):
    #   z_start = index // (self.x_size * self.y_size) + self.slice_start_z
    #   remain  = index % (self.x_size * self.y_size)
    #   x_start = remain // self.y_size
    #   y_start = remain % self.y_size

    #   z_end   = z_start + self.z_out_size
    #   x_end   = x_start + self.x_out_size
    #   y_end   = y_start + self.y_out_size


    #   # random choice from one of sub_dataset
    #   k=np.random.choice(self.im_lb_pair.keys())
    #   im_data = self.im_lb_pair[k]['image']
    #   lb_data = self.im_lb_pair[k]['label']

    #   data    = np.array(im_data[z_start:z_end,x_start:x_end,y_start:y_end]).astype(np.float)
    #   if self.subtract_mean:
    #     data -= 127.0
    #   seg_label   =np.array(lb_data[z_start:z_end,x_start:x_end,y_start:y_end]).astype(np.int)

     
    #   '''set distance larg enough to conver the boundary 
    #      as to put more weight on bouday areas.
    #      which is important when cut distance map for segmentation '''
    #   affineX=affinity(axis=-1,distance =2)
    #   affineY=affinity(axis=-2,distance =2)
    #   affinMap = ((affineX(seg_label)[0] + affineY(seg_label)[0])>0).astype(np.int)


    #   centermap = objCenterMap()
    #   [(x_centerMap, y_centerMap)] = centermap(seg_label)

    #   if self.transform:
    #     data,seg_label,affinMap,x_centerMap,y_centerMap = self.transform(data,seg_label,affinMap,x_centerMap,y_centerMap)

    #   objCenter_map = np.concatenate((x_centerMap,y_centerMap),0) 
      


    #   # if self.transform:
    #   #   data,seg_label,affinMap = self.transform(data,seg_label,affinMap)


      

    #   ''' We compute the runtime obj graidient instead of pre-computed one
    #       to avoid using wrong gradient map when performing data augmentation
    #       such as flip, rotate etc.'''
    #   trans_data_list = self.label_generator(seg_label)
    #   grad_x, grad_y = trans_data_list[0]['gradient']
    #   grad  = np.concatenate((grad_x,grad_y),0)
      



    #   tc_label_dict ={}
    #   for key,value in trans_data_list[0].iteritems():
    #     tc_label_dict[key] = torch.from_numpy(value).float() \
    #                              if key is not 'gradient' \
    #                              else torch.from_numpy(grad).float()

    #   tc_label_dict['affinity']  = torch.from_numpy(affinMap).float()

    #   tc_label_dict['centermap'] = torch.from_numpy(objCenter_map).float()


    #   ''' 
    #   output of tc_label_dict  will have for labels transformed form ground truth 
    #   of segmentatio label:
    #   "affinity", "gradient", "centermap", "dist"
    #   '''
    #   tc_data = torch.from_numpy(data).float()

    #   return tc_data, tc_label_dict

  



    

    def load_data(self):
      
      #data_config = 'conf/cremi_datasets.toml'
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


class labelGenerator(object):
  def __init__(self):
     self.label_generator  = label_transform3D(objSizeMap =True)
  def __call__(self,*input):
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

      output  = []
      for idex,seg_label in enumerate(input):
        affinMap        =  self.affinityFunc(seg_label)
        objCenterMap    =  self.objCenterFunc(seg_label)
        
        trans_data_list =  self.label_generator(seg_label)
        grad_x, grad_y  =  trans_data_list[0]['gradient']
        grad  = np.concatenate((np.expand_dims(grad_x,0), np.expand_dims(grad_y,0)),0)

        tc_label_dict ={}
        for key,value in trans_data_list[0].iteritems():
          tc_label_dict[key] = torch.from_numpy(np.expand_dims(value,0)).float() \
                                 if key is not 'gradient' \
                                 else torch.from_numpy(grad).float()
        tc_label_dict['affinity']  = torch.from_numpy(affinMap).float()
        tc_label_dict['centermap'] = torch.from_numpy(objCenterMap).float()
        
        output.append(tc_label_dict)
      return output
  def output_labels(self):
       ''' output: diction, Key = name of label, value = channel of output '''
       return {'gradient':2,'affinity':1,'centermap':2,'sizemap':1,'distance':1}

  def affinityFunc(self,seg_label):
       affineX=affinity(axis=-1,distance =2)
       affineY=affinity(axis=-2,distance =2)
       affinMap = ((affineX(seg_label)[0] + affineY(seg_label)[0])>0).astype(np.int)
       affinMap = np.expand_dims(affinMap,0)
       return affinMap
  
  def objCenterFunc(self,seg_label):
      centermap = objCenterMap3D()
      [(x_centerMap, y_centerMap)] = centermap(seg_label)
      objCenter_map = np.concatenate((np.expand_dims(x_centerMap,0),np.expand_dims(y_centerMap,0)),0)
      return objCenter_map



def saveGradfiguers(iters,file_prefix,output):
    my_dpi = 96
    plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
    #print ('tc data output shape = {}'.format(output.shape))
    data = output.numpy()
    #print ('output shape = {}'.format(data.shape))
    I = data[0,0,:,:]
    plt.imshow(I)
    plt.savefig(file_prefix +'_readerX{}.png'.format(iters))
    I = data[0,1,:,:]
    plt.imshow(I)
    plt.savefig(file_prefix+'_readerY{}.png'.format(iters))
def saveRawfiguers(iters,file_prefix,output):
    my_dpi = 96
    plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
    data = output.numpy()
    print ('output shape = {}'.format(data.shape))
    I = data[0,:,:]
    plt.imshow(I)
    #pdb.set_trace()
    plt.savefig(file_prefix+'_raw{}.png'.format(iters))


def l2_norm(x):
    #epsilon=torch.cuda.DoubleTensor([1e-12])
    #sq_x   = torch.max(x**2,epsilon)
    #sq_x   = torch.max(x**2,epsilon)
    #e_mat  = torch.zero_like(sq_x)
    sum_x  = torch.sum(x**2,1,keepdim=True)
    sqrt_x = torch.sqrt(sum_x)
    return x/sqrt_x


def compute_angular(x):
    x = F.normalize(x)*0.99999
    #print('x shpe in angu {}'.format(x.shape))
    x_aix = x[:,0,:,:]/torch.sqrt(torch.sum(x**2,1))
    #print('x aix shape {}'.format(x_aix.shape))
    angle_map   = torch.acos(x_aix)
    #print('angle_map shape {}'.format(angle_map.shape))
    return angle_map

    # print ('x shape {}'.format(x.shape))
    # x    = l2_norm(x)*0.9999
    
    # x_aix = x[:,0]/torch.sqrt(torch.sum(x**2,1))
    # angle_map   = torch.acos(x_aix)
    # #pdb.set_trace()
    # return angle_map


     # pred        = pred.transpose(1,2).transpose(2,3).contiguous()
    # gt          = gt.transpose(1,2).transpose(2,3).contiguous()
    # pred        = pred.view(-1, outputChannels)
    # gt          = gt.view(-1, outputChannels)
    # # print(pred[0,:].shape)
    # # s = torch.sqrt(torch.sum((pred*pred),1))
    # # print(s.shape)
    # p_xy        = pred[:,0]/torch.sqrt(torch.sum((pred*pred),1))
    # gt_xy       = gt[:,0]/torch.sqrt(torch.sum((gt*gt),1))
    # err_angle   = torch.acos(p_xy) - torch.acos(gt_xy)
    # loss        = torch.sum(err_angle*err_angle)
    # return loss
def test_angluar_map():
  data_config = '../conf/cremi_datasets_with_tflabels.toml'
  dataset = CRIME_Dataset(data_config = data_config,phase='valid')
  train_loader = DataLoader(dataset=dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=1)
  for i , data in enumerate(train_loader,start =0):
    inputs, labels = data
    labels = labels[:,0,:,:,:]
    ang_map=compute_angular(labels)
    saveRawfiguers(i,'ang_map_test',ang_map)
    if i > 3:
      break



def test_transform():
  data_config = '../conf/cremi_datasets_with_tflabels.toml'
  #data_config = '../conf/cremi_datasets.toml'
  trans=random_transform(VFlip(),HFlip(),Rot90())
  dataset = CRIME_Dataset(data_config   = data_config,phase='valid',transform = None,out_patch_size = (224,224,1),dataset='Set_A')
  train_loader = DataLoader(dataset     = dataset,
                            batch_size  = 1,
                            shuffle     = True,
                            num_workers = 1)
  for i , (inputs,target) in enumerate(train_loader,start =0):
  #for i , all in enumerate(train_loader,start =0):
    
    #labels = labels[:,0,:,:,:]
    #print inputs.shape
    im  = inputs[0,0].numpy()
    #pdb.set_trace()
    #tg1 = target['gradient'][0,0].numpy()
    #tg2 = target['gradient'][0,1].numpy()
    ang_map=compute_angular(target['gradient'])[0].numpy()

    for key, value in target.iteritems():
      print( '{} shape is {}'.format(key,value.shape))

    #print('ang_mp shape = {}'.format(ang_map.shape))
    #print('dist shape {}'.format(target['distance'].shape))
    dist     = target['distance'][0].numpy()
    sizemap  = target['sizemap'][0].numpy()
    affinity = target['affinity'][0].numpy()
    centermap = target['centermap'][0].numpy()
    dist     = np.squeeze(dist)
    sizemap  = np.squeeze(sizemap)
    affinity = np.squeeze(affinity)
    center_x   = np.squeeze(centermap[0])
    center_y   = np.squeeze(centermap[1])
    #print('center shape ={}'.format(center.shape))

    fig,axes = plt.subplots(nrows =2, ncols=3,gridspec_kw = {'wspace':0.01, 'hspace':0.01})
    axes[0,0].imshow(im,cmap='gray')
    axes[0,0].axis('off')
    axes[0,0].margins(0,0)
    
    axes[0,1].imshow(affinity)
    axes[0,1].axis('off')
    axes[0,1].margins(0,0)
    
    axes[1,0].imshow(dist)
    axes[1,0].axis('off')
    axes[1,0].margins(0,0)

    axes[1,1].imshow(np.log(sizemap))
    axes[1,1].axis('off')
    axes[1,1].margins(0,0)


    axes[0,2].imshow(center_x)
    axes[0,2].axis('off')
    axes[0,2].margins(0,0)

    axes[1,2].imshow(center_y)
    axes[1,2].axis('off')
    axes[1,2].margins(0,0)

    plt.margins(x=0.001,y=0.001)
    plt.subplots_adjust(wspace=0, hspace=0)
  
    plt.show()
    #plt.close('all')
    if i >5:
      break


if __name__ == '__main__':
  data_config = '../conf/cremi_datasets.toml'
  dataset = CRIME_Dataset(data_config   = data_config,phase='valid',transform = None,out_patch_size = (2,224,224),sub_dataset='Set_A')