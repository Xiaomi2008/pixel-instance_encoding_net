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
from utils.torch_loss_functions import angularLoss,dice_loss,l2_norm
from utils.printProgressBar import printProgressBar
from label_transform.volumes import Volume
from label_transform.volumes import HDF5Volume
from label_transform.volumes import bounds_generator
from label_transform.volumes import SubvolumeGenerator
#from torch_networks.unet_test import UNet as nUnet
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from utils.EMDataset import CRIME_Dataset
from torch.utils.data import DataLoader
from torch_networks.gcn import GCN
import pdb

class train_test():
    def __init__(self, model, pretrained_model = None,input_size = 224):
        self.input_size = input_size 
        self.model_file = pretrained_model
        self.model_saved_dir   = 'models'
        self.model_save_steps  = 10
        self.model             = model.float()
        self.use_gpu           = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda().float()
        self.use_parallel = False
        self.optimizer    = optim.Adagrad(self.model.parameters(), 
                                            lr=0.001, 
                                            lr_decay=0, 
                                            weight_decay=0)
    def valid(self, dataset):
        dataset.set_phase('valid')
        self.model.eval()
        valid_loader = DataLoader(dataset =dataset,
                                  batch_size=1,
                                  shuffle  =True,
                                  num_workers=1)
        gen = enumerate(valid_loader)
        loss = 0.0
        iters = 100
        save_interval =10
        for i, (data,target) in enumerate(valid_loader, 0):
            target = target[:,0,:,:,:]
            data, target = Variable(data).float(), Variable(target).float()
            if self.use_gpu:
                data = data.cuda().float()
                target =data.cuda().float()
            pred = self.model(data)
            loss += angularLoss(pred, target)
            if i % 10 ==0:
                ang_t_map=compute_angular(target)
                ang_p_map=compute_angular(pred)
                print angle_map.shape
                saveRawfiguers(i,'ang_t_map',ang_t_map)
                saveRawfiguers(i,'ang_p_map',ang_p_map)
                pred_x = pred.data[:,0,:,:]
                pred_y = pred.data[:,1,:,:]
                saveRawfiguers(i,'pred_x',pred_x)
                saveRawfiguers(i,'pred_y',pred_y)
            # del pred
            # del data
            # del target
            if i >= iters-1:
                break
        loss = loss / iters
        print (' valid loss : {}'.format(loss))


    def train(self):
        if not os.path.exists(self.model_saved_dir):
            os.mkdir(self.model_saved_dir)
        if self.model_file:
            self.model.load_state_dict(torch.load(self.model_file))
        gpus = [0]
        use_parallel = True if len(gpus) >1 else False
        if use_parallel:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.model.train()
        dataset = CRIME_Dataset(out_size  = self.input_size,phase = 'train')
        train_loader = DataLoader(dataset =dataset,
                                  batch_size=16,
                                  shuffle  =True,
                                  num_workers=2)
        for epoch in range(5):
            runing_loss = 0.0
            for i, batch in enumerate(train_loader, 0):
                data, target = batch
                target = target[:,0,:,:,:]
                data, target = Variable(data).float(), Variable(target).float()
                
                if self.use_gpu:
                     data   = data.cuda().float()
                     target = target.cuda().float()
               
                self.optimizer.zero_grad()
                output = self.model(data)
                #print('iter {}'.format(i))
                loss = angularLoss(output, target)
                loss.backward()
                #print('done angular and backword')
                self.optimizer.step()
                runing_loss += loss.data[0]
                iter_range = (i+1) // self.model_save_steps
                steps = (i+1) % self.model_save_steps
                start_iters = iter_range*self.model_save_steps
                end_iters   = start_iters + self.model_save_steps
                iters = 'iters : {} to {}:'.format(start_iters,end_iters)
                loss_str  = 0
                if steps == 0:
                    model_save_file = self.model_saved_dir +'/' \
                                  +'{}_size{}_iter_{}.model'.format(self.model.name,self.input_size,i)
                    torch.save(self.model.state_dict(),model_save_file)
                    loss_str = 'loss : {:.5f}'.format(runing_loss/float(self.model_save_steps))
                    printProgressBar(steps, self.model_save_steps, prefix = iters, suffix = loss_str, length = 50)
                    runing_loss = 0.0
                    self.valid(dataset)
                    print(' saved {}'.format(model_save_file))
                    self.model.train()
                else:
                    loss_str = 'loss : {:.5f}'.format(loss.data[0])
                printProgressBar(steps, self.model_save_steps, prefix = iters, suffix = loss_str, length = 50)
               # print('train iter {}, loss = {:.5f}'.format(i,loss.data[0]))
    
    def test(self):
        self.model.eval()
        model.load_state_dict(torch.load(self.model_file))
        dataset = CRIME_Dataset(out_size  = self.input_size, phase ='valid')
        train_loader = DataLoader(dataset =dataset,
                                  batch_size=1,
                                  shuffle  =True,
                                  num_workers=2)
        for i , batch in enumerate(train_loader,start =0):
            data, target = batch
            target = target[:,0,:,:,:]
            data, target = Variable(data).float(), Variable(target).float()
            if self.use_gpu:
                data   = data.cuda().float()
                target = target.cuda().float()

            pred = self.model(data)
            loss = angularLoss(pred, target)
            print('loss:{}'.format(loss))
            ang_t_map=compute_angular(target)
            ang_p_map=compute_angular(pred)
            saveRawfiguers(i,'ang_t_map',ang_t_map)
            saveRawfiguers(i,'ang_p_map',ang_p_map)

            pred_x = pred.data[:,0,:,:]
            pred_y = pred.data[:,1,:,:]

            saveRawfiguers(i,'pred_x',pred_x)
            saveRawfiguers(i,'pred_y',pred_y)
            if i > 5:
                break



# data_config = 'conf/cremi_datasets_with_tflabels.toml'
# volumes = HDF5Volume.from_toml(data_config)
# V_1 = volumes[volumes.keys()[0]]
def compute_angular(x):
    #pdb.set_trace()
    if isinstance(x,Variable):
        x = x.data
    x    = l2_norm(x)*0.9999999999
    x_aix = x/torch.sqrt(torch.sum(x**2,1))
    angle_map   = torch.acos(x_aix)
    return angle_map
def saveRawfiguers(iters,file_prefix,output):
    my_dpi = 96
    plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
    data = output.cpu().numpy()
    print ('output shape = {}'.format(data.shape))
    if data.ndim ==4:
        I = data[0,0]
    elif data.ndim==3:
        I=data[0]
    plt.imshow(I)
    #pdb.set_trace()
    plt.savefig(file_prefix+'{}.png'.format(iters))
def savefiguers(iters,output):
    my_dpi = 96
    plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
    rootdir ='./'
    data = output.data.cpu().numpy()
    print ('output shape = {}'.format(data.shape))
    I = data[0,0,:,:]
    plt.imshow(I)
    plt.savefig('iter_predX_{}.png'.format(iters))
    I = data[0,1,:,:]
    plt.imshow(I)
    plt.savefig('iter_predY_{}.png'.format(iters))
    # I = data
    # vutils.save_image(I[0,0],'iter_predX_{}.png'.format(iters),now)
    # vutils.save_image(I[0,1],'iter_predY_{}.png'.format(iters),now)


# def train(model_file =  None):
#     #use_gpu=torch.cuda.is_available()
#     if not os.path.exists(model_saved_dir):
#         os.mkdir(model_saved_dir)
#     if model_file:
#         netmodel.load_state_dict(torch.load(model_file))
#     gpus = [0]
#     use_parallel = True if len(gpus) >1 else False
#     if use_parallel:
#         #gpus = [0,1,2,3]
#         model = torch.nn.DataParallel(netmodel, device_ids=gpus)
#     else:
#         model = netmodel
#     model.train()
#     optimizer = optim.Adagrad(model.parameters(), lr=0.001, lr_decay=0, weight_decay=0)
#     #im_size =224
#     dataset = CRIME_Dataset(out_size  = input_size)
#     train_loader = DataLoader(dataset =dataset,
#                               batch_size=16,
#                               shuffle  =True,
#                               num_workers=2)
#     for epoch in range(5):
#         runing_loss = 0.0
#         for i, batch in enumerate(train_loader, 0):
#             data, target = batch
#             target = target[:,0,:,:,:]
#             # print ('DataLoader target shape : {}'.format(target.shape))
#             data, target = Variable(data).float(), Variable(target).float()
#             if use_gpu:
#                 data   = data.cuda().float()
#                 target = target.cuda().float()
#             optimizer.zero_grad()
#             output = model(data)
#             loss = angularLoss(output, target)
#             loss.backward()
#             optimizer.step()
#             runing_loss += loss.data[0]
#             if (i+1) % model_save_steps == 0:
#                 model_save_file = model_saved_dir +'/' +'Unet_instance_grad_iter_{}.model'.format(i)
#                 torch.save(model.state_dict(),model_save_file)
#                 print('model saved to {}'.format(model_save_file))
#                 print('[{:5d}] loss: {:.3f}'.format(i,runing_loss/model_save_steps) )
#                 runing_loss = 0

#            print('train iter {}, loss = {:.5f}'.format(i,loss.data[0]))


def test():
    data_config = 'conf/cremi_datasets_with_tflabels.toml'
    volumes = HDF5Volume.from_toml(data_config)
    V_1 = volumes[volumes.keys()[0]]
    model_file = model_saved_dir +'/' +'GCN_instance_grad_iter_{}.model'.format(499)
    netmodel.load_state_dict(torch.load(model_file))
    netmodel.eval()
    im_size =1024
    bounds_gen=bounds_generator(V_1.shape,[1,im_size,im_size])
    sub_vol_gen =SubvolumeGenerator(V_1,bounds_gen)
    for i in xrange(25):
        I = np.zeros([1,1,im_size,im_size])
        C = six.next(sub_vol_gen);
        I[0,0,:,:] = C['image_dataset'].astype(np.int32)
        images = torch.from_numpy(I)
        data = Variable(images).float()
        if use_gpu:
            data=data.cuda().float()
        output = netmodel(data)
        savefiguers(i,output)
def create_model(model_name, input_size =224, pretrained_iter=None):
    model_saved_dir = 'models'
    if model_name == 'GCN':
        model = GCN(num_classes=2, input_size=input_size).float()
    elif model_name == 'Unet':
        model = Unet()

    if  pretrained_iter:
        model_file = model_saved_dir +'/' +'{}_size224_iter_{}.model'.format(model_name,pretrained_iter)
        #model_file =model_saved_dir + '/' + 'GCN_size224_iter49499.model'
        #model_file = model_saved_dir +'/' +'{}_instance_grad_iter_{}.model'.format(model_name,pre_trained_iter)
    else:
        model_file = None
    return model, model_file


if __name__ =='__main__':
    input_size =224
    #model, model_file = create_model('Unet',input_size=input_size,pretrained_iter=69499)
    model, model_file = create_model('GCN',input_size=input_size,pretrained_iter=50499)
    TrTs =train_test(model=model, input_size=input_size,pretrained_model= model_file)
    TrTs.train()
    #TrTs.test()