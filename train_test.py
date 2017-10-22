#import tensorflow as tf
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable
from torch_networks.networks import Unet
from torch_networks.networks import dice_loss as dice_loss
from label_transform.volumes import Volume
from label_transform.volumes import HDF5Volume
from label_transform.volumes import bounds_generator
from label_transform.volumes import SubvolumeGenerator
from torch_networks.unet_test import UNet as nUnet

model = Unet()
#model = nUnet()
model.double()
if torch.cuda.is_available():
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
data_config = 'conf/cremi_datasets_with_tflabels.toml'
volumes = HDF5Volume.from_toml(data_config)
V_1 = volumes[volumes.keys()[0]]

def train():
    model.train()
    im_size =224
    bounds_gen=bounds_generator(V_1.shape,[1,im_size,im_size])
    sub_vol_gen =SubvolumeGenerator(V_1,bounds_gen)
    for i in xrange(200):
        #print ('i == {}'.format(i))
        I = np.zeros([16,1,im_size,im_size])
        T = np.zeros([16,1,im_size,im_size])
        for b in range(16):
            C = six.next(sub_vol_gen);
            print C.keys()
            n_i = C['label_dataset'].astype(np.int32)
            n_l = C['affinityX3_dataset'].astype(np.int32)
            #print(n_i.shape)
            I[b,:,:,:]=n_i
            T[b,:,:,:]=n_i
        labels = torch.from_numpy(I)
        images = torch.from_numpy(T)
        data, target = Variable(images).double(), Variable(labels).double()
        if torch.cuda.is_available():
            data=data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = dice_loss(functional.sigmoid(output), target)
        #loss = functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print loss.data[0]
if __name__ =='__main__':
    train()
