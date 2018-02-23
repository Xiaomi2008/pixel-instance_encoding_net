
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

from transform import VFlip, HFlip, Rot90, random_transform
import torch
from torch.utils.data import Dataset, DataLoader
#from experiment import experiment, experiment_config
from EMDataset import CRIME_Dataset, labelGenerator, CRIME_Dataset3D, labelGenerator3D
from scipy.interpolate import RegularGridInterpolator
#from utils.instance_mask_dataloader \
#    import CRIME_Dataset_3D_labels, instance_mask_GTproc_DataLoader, instance_mask_NNproc_DataLoader
#from torch_networks.networks import Unet, DUnet, MdecoderUnet, Mdecoder2Unet, \
#    MdecoderUnet_withDilatConv, Mdecoder2Unet_withDilatConv, MaskMdecoderUnet_withDilatConv
#from torch_networks.gcn import GCN
#from utils.torch_loss_functions import *
#from utils.printProgressBar import printProgressBar

from torch.autograd import Variable
import time
import torchvision.utils as vutils
#from tensorboardX import SummaryWriter
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation
#import pdb
# from matplotlib import pyplot as plt
#import matplotlib
#from matplotlib import pyplot as plt
import pytoml as toml
import torch.optim as optim
import os
import pdb
class CRIME_Dataset_3D_mask_pair(CRIME_Dataset):
    def __init__(self,
                 in_patch_size =None,
                 out_patch_size=(224, 224, 2),
                 sub_dataset='Set_A',
                 subtract_mean=True,
                 phase='train',
                 transform=None,
                 data_config='conf/cremi_datasets_with_tflabels.toml'):
        super(CRIME_Dataset_3D_mask_pair, self).__init__(sub_dataset=sub_dataset,
                                                      in_patch_size=in_patch_size,
                                                      out_patch_size=out_patch_size,
                                                      subtract_mean=subtract_mean,
                                                      phase=phase,
                                                      transform=transform,
                                                      data_config=data_config)

    def __getitem__(self, index):
        im_data, lb_data = self.random_choice_dataset(self.im_lb_pair)
        data, seg_label  = self.get_random_patch(index, im_data, lb_data)
        #print('1: data shape ={}, seg_label shape ={}'.format(data.shape, seg_label.shape))


        data, seg_mask, lb  = self.__make_slice_MaskPair_with_label__(data, seg_label)
        #if self.subtract_mean:
        #  data -=127.0

        #pdb.set_trace()
        #print('2: data shape ={}, seg_mask shape ={}'.format(data.shape, seg_mask.shape))
        tc_data = torch.from_numpy(np.concatenate([data,seg_mask],0)).float()
        #tc_label_dict = self.gen_label_per_slice(seg_label)
        return tc_data, int(lb)

    def __make_slice_MaskPair_with_label__(self,image, seg_label):
        z_slice = seg_label.shape[0]
        unqiue_ids_in_each_slice = map(lambda x : np.unique(x), seg_label)

        selected_sliceSeg_ids =map(lambda (x,y): select_nonzero_id(x, y), zip(unqiue_ids_in_each_slice, seg_label))

        #[np.unique(seg_label[i]) for i in range(z_slice)]

        #selected_sliceSeg_ids = [select_nonzero_id(unqiue_ids_in_each_seg[i], seg_label[i])
        #                    for i in range(n_samples)
        #                    ]

        masked_slice_idx = np.random.randint(2)
        connect_slice_idx = int(masked_slice_idx ==0)
        masked_seg = (seg_label[masked_slice_idx]== selected_sliceSeg_ids[masked_slice_idx]).astype(np.int)

        #masked_seg_count = np.count_nonzero(masked_seg)
        #print('No of no-zeros in mask seg ={}'.format(masked_seg_count))

        # Give 50% of %70 of chance for connected objects versus non-connected object in two layer of slices
        #if np.random.randint(2) ==0:
        if np.random.rand() > 0.68:
          # find next layer's mask that is connected
          next_layer_mask = (seg_label[connect_slice_idx]== selected_sliceSeg_ids[masked_slice_idx]).astype(np.int)
          if np.sum(next_layer_mask) ==0:
             next_layer_mask = (seg_label[connect_slice_idx]== selected_sliceSeg_ids[connect_slice_idx]).astype(np.int)
             lb_connected =False
          else:
             lb_connected = np.count_nonzero(next_layer_mask)>5
        else:
          # assgin a non-connected mask in nex layer of slice
          #pdb.set_trace()

          struct         = ndimage.generate_binary_structure(2, 3)
          struct        = ndimage.iterate_structure(struct, 3).astype(np.int)

          #print ('mask shape :{}, struct shape:{}'.format(masked_seg.shape,struct2.shape))
          dilate_mask     = ndimage.binary_dilation(masked_seg, structure=struct).astype(masked_seg.dtype)
          bool_mask       = dilate_mask.astype(np.bool)
          nearby_idx      = np.unique(seg_label[connect_slice_idx][bool_mask])
          sel_idx         = select_nonzero_id(nearby_idx,seg_label[connect_slice_idx])
          next_layer_mask = (seg_label[connect_slice_idx]== sel_idx).astype(np.int)
          lb_connected    = (sel_idx == selected_sliceSeg_ids[masked_slice_idx]) \
                             and np.count_nonzero(next_layer_mask) >10




        mask = np.stack([masked_seg,next_layer_mask],0) \
               if masked_slice_idx ==0  else np.stack([next_layer_mask,masked_seg],0)


        out_size = [self.x_out_size,self.y_out_size]

        #print('image shape before crop = {}'.format(image.shape))
        image, mask=crop_and_resize(image,mask,out_size=out_size)

        #print('image shape after crop = {}'.format(image.shape))

        return image, mask, lb_connected


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



def crop_and_resize(image, mask, out_size=[224,224]):
     assert image.shape[0]==2
     assert mask.shape[0] ==2
     ims_in = image.copy()
     ims,msks=crop_image_from_two_adjecent_masks(ims_in,mask)
     #print('crop_imag_two after crop = {}'.format(ims.shape))
     if ims.shape[0] ==0 or ims.shape[1] ==0:
      pdb.set_trace()
     img=np.stack([resize_img(im,out_size, mode ='linear') for im in ims],0)
     msk=np.stack([resize_img(im,out_size, mode ='linear') for im in msks],0)
     return img,msk


def crop_image_from_two_adjecent_masks(img,mask):
    def bbox(img):
      #rows = np.any(img, axis=1)
      #cols = np.any(img, axis=0)
      rows = np.any(img, axis=0)
      cols = np.any(img, axis=1)
      try:
        ymin, ymax = np.where(rows)[0][[0, -1]]
      except:
        pdb.set_trace()
      xmin, xmax = np.where(cols)[0][[0, -1]]
      return (xmin, xmax), (ymin,ymax)

    (xmin1, xmax1), (ymin1,ymax1) =bbox(mask[0])
    (xmin2, xmax2), (ymin2,ymax2) =bbox(mask[1])
    xmin = min(xmin1,xmin2)
    xmax = max(xmax1,xmax2)
    ymin = min(ymin1,ymin2)
    ymax = max(ymax1,ymax2)

    '''Try to enlarge the view so that net can see larger area that conver not only
       two possible adjecent object but also the peripheral to give net more information
       for classification '''
    im_w  =img.shape[1]
    im_h = img.shape[2]

    width = xmax -xmin
    height =ymax -ymin
    enlarge_view_scale = 2.0


    

    xmin = max(0,int((xmin + width /2.0) - (width*enlarge_view_scale)/2.0))
    xmax = min(im_w,int((xmin + width /2.0) + (width*enlarge_view_scale)/2.0))

    ymin = max(0,int((ymin + height /2.0) - (height*enlarge_view_scale)/2.0))
    ymax = min(im_h,int((ymin + height /2.0) + (height*enlarge_view_scale)/2.0))

    # im1 =img[0,xmin:xmax,ymin:ymax]
    # im2 =img[1,xmin:xmax,ymin:ymax]
    # msk1 =mask[0,xmin:xmax,ymin:ymax]
    # msk2 =mask[1,xmin:xmax,ymin:ymax]

    #im =np.stack([im1,im2],axis =0)
    #msk=np.stack([msk1,msk2],axis =0)

    if xmax-xmin <=0 or ymax-ymin <=0:
      #pdb.set_trace( )
      print('width =0 or height = o in mark')

    if xmax-xmin <=0:
      xmin =10
      xmax =20

    if ymax-ymin <=0:
      ymin =10
      ymax =20


    im = img[:,xmin:xmax,ymin:ymax].copy()
    msk= mask[:,xmin:xmax,ymin:ymax].copy()

    return im,msk





def resize_img(img,out_size,mode ='linear'):
    x_out_size,y_out_size = out_size[0],out_size[1]
    x_in_size,y_in_size = img.shape
    assert x_in_size >0
    assert y_in_size >0
    x = np.linspace(0,x_in_size-1, x_in_size)
    y = np.linspace(0,y_in_size-1, y_in_size)
    img_interp = RegularGridInterpolator((x,y), img)
    xi = np.linspace(0,x_in_size-1,x_out_size)
    yi = np.linspace(0,y_in_size-1,y_out_size)
    pts  = np.squeeze(np.array(zip(np.meshgrid(xi,yi, indexing='ij'))))  
    img_s = img_interp(pts.T, method =mode)

    return img_s
  
def select_nonzero_id(unique_ids, seg, threshed=30):
    islarger = False
    count = 0
    while not islarger and count <100:
        count +=1
        sid = np.random.choice(unique_ids)
        if sid > 0:
            islarger = np.sum((seg == sid).astype(np.int)) > threshed
    return sid

def test_transform():
    data_config = 'conf/cremi_datasets_with_tflabels.toml'
    # data_config = '../conf/cremi_datasets.toml'
    trans = random_transform([VFlip(), HFlip(), Rot90()])
    dataset = CRIME_Dataset_3D_mask_pair(data_config=data_config, 
                                         phase='train',
                                         in_patch_size=(640,640,2),
                                         out_patch_size=(224, 224, 2),
                                         transform=trans)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=10,
                              shuffle=True,
                              num_workers=1)

    # for i in range(100):
    #   tc_data,lb=dataset.__getitem__(i)

    for i, (data, lb) in enumerate(train_loader, start=0):
      #print('iter {} shape = {}, lb ={}'.format(i,data.shape,lb))
      for j in range(len(data)):
        visualize_test(data[j,0:2,:],data[j,2:,:],lb[j],j)
      if i >30:
        break

def visualize_test(im,seg_mask,lb ,idx):
  fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
  axes[0, 0].imshow(im[0], cmap='gray')
  axes[0, 0].axis('off')
  axes[0, 0].margins(0, 0)
  title_str = 'same object = {}'.format(lb)
  axes[0,0].set_title(title_str)

  axes[0, 1].imshow(im[1], cmap='gray')
  axes[0, 1].axis('off')
  axes[0, 1].margins(0, 0)

  axes[1, 0].imshow(seg_mask[0])
  axes[1, 0].axis('off')
  axes[1, 0].margins(0, 0)

  axes[1, 1].imshow(seg_mask[1])
  axes[1, 1].axis('off')
  axes[1, 1].margins(0, 0)

  plt.savefig('mask_test{}.png'.format(idx))

if __name__ =='__main__':
  test_transform()


