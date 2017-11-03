import os, sys
sys.path.append('../')
import pdb
import torch
import numpy as np
from transform import *
import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from label_transform.volumes import Volume
from label_transform.volumes import HDF5Volume
from label_transform.volumes import bounds_generator
from label_transform.volumes import SubvolumeGenerator


class CRIME_Dataset(Dataset):
    """ EM dataset."""

    # Initialize EM data
    def __init__(self, out_size = 224, 
                 dataset = 'Set_A',
                 subtract_mean = False,
                 phase   = 'train',
                 transform = None,
                 data_config = 'conf/cremi_datasets_with_tflabels.toml'):
      self.dataset      = dataset
      self.phase        = phase
      self.x_out_size   = out_size
      self.y_out_size   = out_size
      self.z_out_size   = 1
      self.data_config  = data_config
      self.subtract_mean = subtract_mean
      self.transform = transform
      self.load_hdf()

      dim_shape         = self.im_data.shape
      self.y_size       = dim_shape[2] -self.x_out_size + 1
      self.x_size       = dim_shape[1] -self.y_out_size + 1 
      self.set_phase(phase)
      #self.z_size       = dim_shape[0] -self.z_out_size + 1
    def set_phase(self,phase):
      self.phase = phase
      if phase == 'train':
        self.slice_start_z= 0
        self.slice_end_z   = 99
      elif phase == 'valid':
        self.slice_start_z = 100
        self.slice_end_z = 124

      self.z_size = self.slice_end_z - self.slice_end_z +1


        
    def __getitem__(self, index):
      z_start = index // (self.x_size * self.y_size) + self.slice_start_z
      remain  = index % (self.x_size * self.y_size)
      x_start = remain // self.y_size
      y_start = remain % self.y_size

      z_end   = z_start + self.z_out_size
      x_end   = x_start + self.x_out_size
      y_end   = y_start + self.y_out_size

      data    = np.array(self.im_data[z_start:z_end,x_start:x_end,y_start:y_end]).astype(np.float)
      if self.subtract_mean:
        data -= 127.0
      target_ch1  =np.array(self.gradX[z_start:z_end,x_start:x_end,y_start:y_end])
      target_ch2  =np.array(self.gradY[z_start:z_end,x_start:x_end,y_start:y_end])


      if self.transform:
        data, target_ch1, target_ch2 = self.transform(data,target_ch1,target_ch2)
        #print data.shape, target_ch1.shape, target_ch2.shape
        #print len(out)

      
      # (data, target_ch1, target_ch2) = out
      #target_ch1  = np.expand_dims(target_ch1,1)
      #target_ch2  = np.expand_dims(target_ch2,1)
      target      = np.concatenate((target_ch1,target_ch2),0)
      #print data.shape, target.shape
      #print('target_xy shape:{}'.format(target.shape))
      #pdb_set_trace()

      #target  = self.gt_data[z_start:z_end,x_start:x_end,y_start:y_end]
      tc_data, tc_target= torch.from_numpy(data).float(), torch.from_numpy(target).float()

      #print('TC_target_xy shape:{}'.format(tc_target.shape))

      return tc_data,tc_target

      #return self.im_data[z_start:z_end,x_start:x_end,y_start:y_end], \
      #       self.lb_data[z_start:z_end,x_start:x_end,y_start:y_end]
      # return self.im_data[z_start:z_end,x_start:x_end,y_start:y_end], \
      #        self.gradient_data[z_start:z_end,x_start:x_end,y_start:y_end]



    def __len__(self):
      self.len = self.x_size * self.y_size * self.z_size 
      return self.len

    def load_hdf(self):
      
      #data_config = 'conf/cremi_datasets.toml'
      volumes = HDF5Volume.from_toml(self.data_config)
      #data_name ={'Set_A':'Sample A','Set_B':'Sample B','Set_C':'Sample C'}
      data_name = {'Set_A':'Sample A with extra transformed labels'}
      #data_name = {'Set_A':'Sample A'}
      self.V = volumes[data_name[self.dataset]]
      self.gradX = self.V.data_dict['gradX_dataset']
      self.gradY = self.V.data_dict['gradY_dataset']
      self.lb_data = self.V.data_dict['label_dataset']
      self.im_data = self.V.data_dict['image_dataset']
      #self.gradX = self.lb_data
      #self.gradY = self.lb_data
      

      # gradX = np.expand_dims(gradX, 1)
      # gradY = np.expand_dims(gradY, 1)
      # self.gd_data = np.concatenate((gradX,gradY),axis=1)
      # self.lb_data = np.array(self.V.data_dict['label_dataset']).astype(np.int32)
      # self.im_data = np.array(self.V.data_dict['image_dataset']).astype(np.int32)

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
    x    = l2_norm(x)*0.9999
    
    x_aix = x[:,0]/torch.sqrt(torch.sum(x**2,1))
    angle_map   = torch.acos(x_aix)
    #pdb.set_trace()
    return angle_map


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
  trans=random_transform(VFlip(),HFlip(),Rot90())
  dataset = CRIME_Dataset(data_config = data_config,phase='valid',transform = trans)
  train_loader = DataLoader(dataset=dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=1)
  for i , (inputs,labels) in enumerate(train_loader,start =0):
    #labels = labels[:,0,:,:,:]
    im = inputs[0,0].numpy()
    tg = labels[0,0].numpy()
    #print(im)
    im = (im - np.min(im))/np.mean(im)
    tg = (tg - np.min(tg))/np.mean(tg)
    #im*=255
    tg*=255
    cv2.imshow("image",im)
    cv2.imshow("lb",tg)
    cv2.waitKey(2000)
    if i >20:
      break


if __name__ == '__main__':
  test_transform()
  #test_angluar_map()
  # data_config = '../conf/cremi_datasets_with_tflabels.toml'
  # dataset = CRIME_Dataset(data_config = data_config)
  # train_loader = DataLoader(dataset=dataset,
  #                         batch_size=1,
  #                         shuffle=True,
  #                         num_workers=1)
  # for epoch in range(1):
  #   #d,l = dataset.__getitem__(1000)
  #   for i, data in enumerate(train_loader, start=0):
  #        # get the inputs
  #        inputs, labels = data
  #        labels = labels[:,0,:,:,:]
  #        pdb.set_trace()
  #        print ('iter = {} shape labels = {}'.format(i,labels.shape))
  #        saveGradfiguers(i,'grad',labels,)
  #        #saveRawfiguers(i,'raw',inputs,)
  #        if i > 3:
  #         break
