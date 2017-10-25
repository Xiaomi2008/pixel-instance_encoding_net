#import tensorflow as tf
import os
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
#from torch_networks.unet_test import UNet as nUnet
from matplotlib import pyplot as plt
from utils.EMDataset import CRIME_Dataset
from torch.utils.data import DataLoader
import pdb


model_saved_dir = 'models'
model_save_steps = 500
model = Unet().double()
#model = nUnet()
#model.float()
use_gpu=torch.cuda.is_available()
#use_gpu=False
use_parallel = False
if use_gpu:
    model.cuda().double()
    if use_parallel:
        #gpus = [0,1,2,3]
        gpus =[0,1]
        model = torch.nn.DataParallel(model, device_ids=gpus)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
# data_config = 'conf/cremi_datasets_with_tflabels.toml'
# volumes = HDF5Volume.from_toml(data_config)
# V_1 = volumes[volumes.keys()[0]]

def savefiguers(iters,output):
    rootdir ='./'
    data = output.data.cpu().numpy()
    print ('output shape = {}'.format(data.shape))
    I = data[0,0,:,:]
    plt.imshow(I)
    plt.savefig('iter_predX_{}.png'.format(iters))
    I = data[0,1,:,:]
    plt.imshow(I)
    plt.savefig('iter_predY_{}.png'.format(iters))


def train():
    #use_gpu=torch.cuda.is_available()
    if not os.path.exists(model_saved_dir):
        os.mkdir(model_saved_dir)
    model.train()
    im_size =224
    dataset = CRIME_Dataset(out_size  = im_size)
    train_loader = DataLoader(dataset =dataset,
                              batch_size=16,
                              shuffle  =True,
                              num_workers=2)
    for epoch in range(1):
        for i, batch in enumerate(train_loader, 0):
            data, target = batch
            data, target = Variable(data).double(), Variable(target).double()
            if use_gpu:
                data   = data.cuda().double()
                target = target.cuda().double()
            optimizer.zero_grad()
            output = model(data)
            loss = angularLoss(output, target)
            loss.backward()
            optimizer.step()

            if i+1 % model_save_steps == 0:
                model_save_file = model_saved_dir +'/' +'Unet_instance_grad_iter_{}.model'.format(i)
                torch.save(model.state_dict(),model_save_file)

            print('iter {}, loss = {:.5f}'.format(i,loss.data[0]))

        # print ('Input iter = {} shape inputs = {}'.format(i,inputs.shape))


    # bounds_gen=bounds_generator(V_1.shape,[1,im_size,im_size])
    # sub_vol_gen =SubvolumeGenerator(V_1,bounds_gen)
    # for i in xrange(500):
    #     #print ('i == {}'.format(i))

    #     I = np.zeros([16,1,im_size,im_size])
    #     T = np.zeros([16,2,im_size,im_size])
    #     for b in range(16):
    #         C = six.next(sub_vol_gen);
    #         I[b,:,:,:]= C['image_dataset'].astype(np.int32)
    #         T[b,0,:,:]= C['gradX_dataset'].astype(np.double)
    #         T[b,1,:,:]= C['gradY_dataset'].astype(np.double)
    #     images = torch.from_numpy(I)
    #     labels = torch.from_numpy(T)
    #     data, target = Variable(images).double(), Variable(labels).double()
    #     if use_gpu:
    #         data=data.cuda().double()
    #         target = target.cuda().double()
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = angularLoss(output, target)
    #     loss.backward()
    #     optimizer.step()
    #     print('iter {}, loss = {:.5f}'.format(i,loss.data[0]))

        #if i % 100 ==0:
        #    test(iters=i)
        #print loss.data[0]
def validate():
    model.eval()

def test():
    data_config = 'conf/cremi_datasets.toml'
    volumes = HDF5Volume.from_toml(data_config)
    V_1 = volumes[volumes.keys()[0]]
    model_file = model_saved_dir +'/' +'Unet_instance_grad_iter_{}.model'.format(5000)
    model.eval()
    im_size =1024
    bounds_gen=bounds_generator(V_1.shape,[1,im_size,im_size])
    sub_vol_gen =SubvolumeGenerator(V_1,bounds_gen)
    for i in xrange(10):
        I = np.zeros([1,1,im_size,im_size])
        C = six.next(sub_vol_gen);
        I[0,0,:,:] = C['image_dataset'].astype(np.int32)
        images = torch.from_numpy(I)
        data = Variable(images).double()
        if use_gpu:
            data=data.cuda().double()
        output = model(data)
        savefiguers(i,output)
if __name__ =='__main__':
    #train()
    test()
