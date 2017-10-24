import os, sys
sys.path.append('../')
import numpy as np
from torch.utils.data import Dataset, DataLoader
from label_transform.volumes import Volume
from label_transform.volumes import HDF5Volume
from label_transform.volumes import bounds_generator
from label_transform.volumes import SubvolumeGenerator

class CRIME_Dataset(Dataset):
    """ EM dataset."""

    # Initialize your data
    def __init__(self, out_size = 224, dataset = 'Set_A'):
      self.dataset      = dataset
      self.x_out_size   = out_size
      self.y_out_size   = out_size
      self.z_out_size   = 1
      self.load_hdf()
        
    def __getitem__(self, index):
      z_start = index / (self.x_size * self.y_size)
      remain  = index % (self.x_size * self.y_size)
      x_start = remain / self.y_size
      y_start = remain % self.y_size

      z_end   = z_start + self.z_size
      x_end   = x_start + self.x_size
      y_end   = y_start + self.y_size


      return self.im_data[z_start:z_end,x_start:x_end,y_start:y_end], \
             self.lb_data[z_start:z_end,x_start:x_end,y_start:y_end]
      # return self.im_data[z_start:z_end,x_start:x_end,y_start:y_end], \
      #        self.gradient_data[z_start:z_end,x_start:x_end,y_start:y_end]



    def __len__(self):
      dim_shape = self.im_data.shape
      self.y_size  = dim_shape[2] -self.out_size + 1
      self.x_size  = dim_shape[1] -self.out_size + 1 
      self.z_size  = dim_shape[0]
      self.len = x_size * y_size * z_size 
      return self.len

    def load_hdf(self):
      # data_config = '../conf/cremi_datasets_with_tflabels.toml'
      data_config = '../conf/cremi_datasets.toml'
      volumes = HDF5Volume.from_toml(data_config)
      data_name ={'Set_A':'Sample A','Set_B':'Sample B','Set_C':'Sample C'}
      # data_name = {'Set_A':'Sampe A with extra transformed labels'}
      self.V = volumes[data_name[self.dataset]]
      # gradX = np.array(self.V['gradX_dataset']).astype(np.int32)
      # gradY = np.array(self.V['gradY_dataset']).astype(np.int32)
      # self.gradient_data = np.concatenate(gradX,gradY,axis=-1)
      self.lb_data = np.array(self.V.data_dict['label_dataset']).astype(np.int32)
      self.im_data = np.array(self.V.data_dict['image_dataset']).astype(np.int32)


if __name__ == '__main__':
  dataset = CRIME_Dataset()
  train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
  for epoch in range(1):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        print ('Input iter = {} shape inputs = {}'.format(i,inputs.shape))
