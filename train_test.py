#import tensorflow as tf
import os
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.nn.modules.loss import MSELoss
#import torch.nn.MSELoss
from torch.autograd import Variable
from torch_networks.networks import Unet
from torch_networks.duc  import ResNetDUCHDC
from torch_networks.gcn import GCN
from torch_networks.unet2 import UNet as Unet2
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
import time
import pdb

class train_test():
    def __init__(self, model, pretrained_model = None,input_size = 224):
        self.input_size = input_size 
        self.model_file = pretrained_model
        self.model_saved_dir   = 'models'
        self.model_save_steps  = 500
        self.model             = model.float()
        self.use_gpu           = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
        self.use_parallel = False
        self.mse_loss = torch.nn.MSELoss()
        self.optimizer    = optim.Adagrad(self.model.parameters(), 
                                            lr=0.001, 
                                            lr_decay=0, 
                                            weight_decay=0)
        subtract_mean = False if model.name is 'Unet' else True
        self.trainDataset = CRIME_Dataset(out_size  = self.input_size, phase = 'train',subtract_mean =subtract_mean)
        self.validDataset = CRIME_Dataset(out_size  = self.input_size, phase = 'valid',subtract_mean =subtract_mean)
        if self.model_file:
            self.model.load_state_dict(torch.load(self.model_file))
    def valid(self):
        dataset = self.validDataset
        dataset.set_phase('valid')
        self.model.eval()
        valid_loader = DataLoader(dataset =dataset,
                                  batch_size=1,
                                  shuffle  =True,
                                  num_workers=1)
        loss = 0.0
        iters = 20
        save_interval =20
        for i, (data,target) in enumerate(valid_loader, 0):
            target = target[:,0,:,:,:]
            #print(target)
            data, target = Variable(data).float(), Variable(target).float()
            if self.use_gpu:
                data = data.cuda().float()
                target =target.cuda().float()
            pred = self.model(data)
            loss += angularLoss(pred, target).data[0]
            #print('t shape = {}'.format(target.data.shape))
            #print('p shape = {}'.format(pred.data.shape))
            if i % iters ==0:
                ang_t_map=compute_angular(target)
                ang_p_map=compute_angular(pred)
                model_name=self.model.name
                saveRawfiguers(i,'ang_t_map_'+model_name,ang_t_map)
                saveRawfiguers(i,'ang_p_map_'+model_name,ang_p_map)
                pred_x = pred.data[:,0,:,:]
                pred_y = pred.data[:,1,:,:]
                saveRawfiguers(i,'pred_x_'+model_name,pred_x)
                saveRawfiguers(i,'pred_y_'+model_name,pred_y)
            if i >= iters-1:
                break
        print (loss)
        loss = loss / iters
        self.model.train()
        print (' valid loss : {:.3f}'.format(loss))

    def train(self):
        if not os.path.exists(self.model_saved_dir):
            os.mkdir(self.model_saved_dir)
        gpus = [0]
        use_parallel = True if len(gpus) >1 else False
        num_workers =2
        if use_parallel:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
            num_workers =2
        self.model.train()
        #dataset = CRIME_Dataset(out_size  = self.input_size,phase = 'train')
        dataset  = self.trainDataset
        train_loader = DataLoader(dataset =dataset,
                                  batch_size=16,
                                  shuffle  =True,
                                  num_workers=num_workers)
        for epoch in range(5):
            runing_loss = 0.0
            start_time = time.time()
            for i, (data,target) in enumerate(train_loader, 0):
                target = target[:,0,:,:,:]
                data, target = Variable(data).float(), Variable(target).float()
                  
                if self.use_gpu:
                     data   = data.cuda().float()
                     target = target.cuda().float()
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = angularLoss(output, target)
                #loss = self.mse_loss(output,target)
                #loss = 0.9*a_loss+0.1*m_loss
                loss.backward()
                self.optimizer.step()
                

                runing_loss += loss.data[0]
                iter_range = (i+1) // self.model_save_steps
                steps = (i+1) % self.model_save_steps
                start_iters = iter_range*self.model_save_steps
                end_iters   = start_iters + self.model_save_steps
                iters = 'iters : {} to {}:'.format(start_iters,end_iters)
                loss_str  = 0
                elaps_time =time.time() - start_time
                if steps == 0:
                    model_save_file = self.model_saved_dir +'/' \
                                  +'{}_size{}_iter_{}.model'.format(self.model.name,self.input_size,i)
                    torch.save(self.model.state_dict(),model_save_file)
                    loss_str = 'loss : {:.5f}'.format(runing_loss/float(self.model_save_steps))
                    printProgressBar(self.model_save_steps, self.model_save_steps, prefix = iters, suffix = loss_str, length = 50)
                    runing_loss = 0.0
                    self.valid()
                    
                    print(' saved {}'.format(model_save_file))
                    start_time = time.time()
                else:
                    loss_str = 'loss : {:.3f}'.format(loss.data[0])
                loss_str =  loss_str + ', time : {:.2}s'.format(elaps_time)
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
            print('loss:{}'.format(loss.data[0]))
            ang_t_map=compute_angular(target)
            ang_p_map=compute_angular(pred)
            saveRawfiguers(i,'ang_t_map_{}'.format(self.model.name),ang_t_map)
            saveRawfiguers(i,'ang_p_map_{}'.format(self.model.name),ang_p_map)
            pred_x = pred.data[:,0,:,:]
            pred_y = pred.data[:,1,:,:]
            saveRawfiguers(i,'pred_x_{}'.format(self.model.name),pred_x)
            saveRawfiguers(i,'pred_y_{}'.format(self.model.name),pred_y)
            if i > 5:
                break



# data_config = 'conf/cremi_datasets_with_tflabels.toml'
# volumes = HDF5Volume.from_toml(data_config)
# V_1 = volumes[volumes.keys()[0]]
def compute_angular(x):
    # input x must be a 4D data [n,c,h,w]
    if isinstance(x,Variable):
        x = x.data
   # x    = l2_norm(x)*0.999999
    x = F.normalize(x)*0.99999
    #print(x.shape)
    x_aix = x[:,0,:,:]/torch.sqrt(torch.sum(x**2,1))
    angle_map   = torch.acos(x_aix)
    return angle_map

def saveRawfiguers(iters,file_prefix,output):
    from torchvision.utils import save_image
    my_dpi = 96
    plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
    data = output.cpu().numpy()
    #data = output.cpu()
    #print ('output shape = {}'.format(data.shape))
    if data.ndim ==4:
         I = data[0,0]
    elif data.ndim==3:
         I=data[0]
    else:
         I = data
    #save_image(data,file_prefix+'{}.png'.format(iters))
    plt.imshow(I)
    #pdb.set_trace()
    plt.savefig(file_prefix+'{}.png'.format(iters))
    plt.close('all')

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
        model = GCN(num_classes=2, input_size=input_size)
    elif model_name == 'Unet':
        model = Unet()
    elif model_name == 'Unet2':
        model = Unet2(num_classes=2)
    elif model_name == 'Unet2DeformConv':
        model = Unet2(num_classes=2,deformConv=True)
    elif model_name == 'DUCHDC':
        model =ResNetDUCHDC(num_classes=2)

    if  pretrained_iter:
        model_file = model_saved_dir +'/' +'{}_size224_iter_{}.model'.format(model_name,pretrained_iter)
        #model_file =model_saved_dir + '/' + 'GCN_size224_iter49499.model'
        #model_file = model_saved_dir +'/' +'{}_instance_grad_iter_{}.model'.format(model_name,pre_trained_iter)
    else:
        model_file = None
    return model, model_file


if __name__ =='__main__':
    input_size =320
    #model, model_file = create_model('Unet',input_size=input_size,pretrained_iter=5499)
    #model, model_file = create_model('Unet2',input_size=input_size)

    model, model_file = create_model('Unet2DeformConv',input_size=input_size)
    #model, model_file = create_model('GCN',input_size=input_size,pretrained_iter=13999)
    #model, model_file = create_model('DUCHDC',input_size = input_size)
    TrTs =train_test(model=model, input_size=input_size,pretrained_model= model_file)
    TrTs.train()
    #TrTs.test()