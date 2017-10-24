#import tensorflow as tf
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable
from torch_networks.networks import Unet
from torch_networks.networks import dice_loss as dice_loss, angularLoss
from label_transform.volumes import Volume
from label_transform.volumes import HDF5Volume
from label_transform.volumes import bounds_generator
from label_transform.volumes import SubvolumeGenerator
from torch_networks.unet_test import UNet as nUnet
from matplotlib import pyplot as plt

model = Unet().double()
#model = nUnet()
#model.float()
use_gpu=torch.cuda.is_available()
if use_gpu:
    model.cuda().double()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
data_config = 'conf/cremi_datasets_with_tflabels.toml'
volumes = HDF5Volume.from_toml(data_config)
V_1 = volumes[volumes.keys()[0]]
def savefiguers(iter,output):
    rootdir ='./'
    plt.imshow(output.data.numpy()[0])
    plt.savefigure('iter_image{}.png'.format(iter))


def train():
    #use_gpu=torch.cuda.is_available()
   
    model.train()
    im_size =224
    bounds_gen=bounds_generator(V_1.shape,[1,im_size,im_size])
    sub_vol_gen =SubvolumeGenerator(V_1,bounds_gen)
    for i in xrange(1000):
        #print ('i == {}'.format(i))

        I = np.zeros([16,1,im_size,im_size])
        T = np.zeros([16,2,im_size,im_size])
        for b in range(16):
            C = six.next(sub_vol_gen);
            I[b,:,:,:]= C['label_dataset'].astype(np.int32)
            T[b,0,:,:]= C['gradX_dataset'].astype(np.double)
            T[b,1,:,:]= C['gradY_dataset'].astype(np.double)
        images = torch.from_numpy(I)
        labels = torch.from_numpy(T)
        data, target = Variable(images).double(), Variable(labels).double()
        if use_gpu:
            data=data.cuda().double()
            target = target.cuda().double()
        optimizer.zero_grad()
        output = model(data)
        loss = angularLoss(output, target)
        loss.backward()
        optimizer.step()
        print('iter {}, loss = {:.5f}'.format(i,loss.data[0]))

        if i % 100 ==0:
            test(iters=i)
        #print loss.data[0]


def test(iters = 0):
    use_gpu=torch.cuda.is_available()
    model.eval()
    im_size =1024
    bounds_gen=bounds_generator(V_1.shape,[1,im_size,im_size])
    sub_vol_gen =SubvolumeGenerator(V_1,bounds_gen)
    for i in xrange(10):
        I = np.zeros([1,1,im_size,im_size])
        T = np.zeros([1,2,im_size,im_size])
        for b in range(1):
            C = six.next(sub_vol_gen);
            n_i = C['label_dataset'].astype(np.int32)
            affin_x3 = C['gradX_dataset'].astype(np.float)
            affin_y3 = C['gradY_dataset'].astype(np.float)
            I[b,:,:,:]=n_i
            T[b,0,:,:]=affin_x3
            T[b,1,:,:]=affin_y3
        images = torch.from_numpy(I)
        labels = torch.from_numpy(T)
        data, target = Variable(images).double(), Variable(labels).double()
        if use_gpu:
            data=data.cuda().double()
            target = target.cuda().double()
        output = model(data)
        savefiguers(iters,output)
        model.train()

if __name__ =='__main__':
    train()
    test()
