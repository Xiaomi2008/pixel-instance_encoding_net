
import torch
from torch_networks.networks import Unet,DUnet, MdecoderUnet
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch_networks.networks import Unet,DUnet,MdecoderUnet,Mdecoder2Unet,MdecoderUnet_withDilatConv
# from transform import *

from utils.EMDataset import CRIME_Dataset,labelGenerator
from utils.transform import VFlip, HFlip, Rot90, random_transform
from utils.torch_loss_functions import *
from utils.printProgressBar import printProgressBar
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import numpy as np
#from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pytoml as toml
import torch.optim as optim
import time
import pdb
import os


class experiment_config():
  def __init__(self, config_file):
    self.parse_toml(config_file)

    print self.label_conf['labels']
    print '_'.join(self.label_conf['labels'])
    networks = \
              {'Unet' : Unet,'DUnet' : DUnet,'MDUnet': MdecoderUnet, \
              'MDUnetDilat':MdecoderUnet_withDilatConv, 'M2DUnet':Mdecoder2Unet}
   
    self.data_transform     = self.data_Transform(self.data_aug_conf['transform'])
    self.label_generator    = self.label_Generator()
    
    self.train_dataset, self.valid_dataset \
    = self.dataset(self.net_conf['patch_size'],self.data_transform)

    #labels_in_use = ['gradient','sizemap','affinity','centermap','distance']
    label_in_use = self.label_conf['labels']

    label_ch_pair ={}

    # data_out_labels is a dict that stores "label name" as key and "# of channel for that label" as value
    data_out_labels = self.train_dataset.output_labels()


    for lb in label_in_use:
      label_ch_pair[lb] =  data_out_labels[lb]
   

    in_ch =self.net_conf['patch_size'][2]

    if 'sub_net' in self.conf:
      subnet_model = networks[self.conf['sub_net']['model']]
      self.sub_network = subnet_model(target_label = label_ch_pair, in_ch = in_ch)

      
      net_model    = networks[self.net_conf['model']]
      self.network = net_model(self.sub_network, freeze_net1 = self.conf['sub_net']['freeze_weight'])
    else:
      net_model= networks[self.net_conf['model']]
      self.network = net_model(target_label = label_ch_pair,in_ch = in_ch)

  def parse_toml(self,file):
      with open(file, 'rb') as fi:
            conf = toml.load(fi)
            net_conf  = conf['network']
            net_conf['model']            = net_conf.get('model','DUnet')
            net_conf['model_saved_dir']  = net_conf.get('model_saved_dir','model')
            net_conf['load_train_iter']  = net_conf.get('load_train_iter', None)
            net_conf['model_save_steps'] = net_conf.get('model_save_steps',500)
            net_conf['patch_size']       = net_conf.get('patch_size',[320,320,1])
            #net_conf['learning_rate']    = net_conf.get('learning_rate',0.01)


            train_conf = conf['train']
            train_conf['final_loss_only']  = train_conf.get('final_loss_only',False)
            train_conf['learning_rate']    = train_conf.get('learning_rate',0.01)
            #net_conf['trained_file']     = net_conf.get('trained_file','')
           
            label_conf =conf['target_labels']


            label_conf['labels']=label_conf.get('labels',['gradient','sizemap','affinity','centermap','distance'])
            label_conf['final_label']=label_conf.get('final_labels','distance')
            data_aug_conf = conf['data_augmentation']

            #print data_aug_conf
            data_aug_conf['transform'] = data_aug_conf.get('transform',['vflip','hflip','rot90'])

            self.label_conf        = label_conf
            self.data_aug_conf     = data_aug_conf
            self.net_conf          = net_conf
            self.dataset_conf      = conf['dataset']
            self.train_conf        = train_conf
            self.conf              = conf



  def dataset(self,out_patch_size,transform):
      sub_dataset = self.dataset_conf['sub_dataset']
      out_patch_size = self.net_conf['patch_size']
      print 'this out {}'.format(out_patch_size)
      train_dataset = CRIME_Dataset(out_patch_size   =  out_patch_size, 
                                    phase            =  'train',
                                    subtract_mean    =  True,
                                    transform        =  self.data_transform,
                                    sub_dataset      =  sub_dataset)
          
      valid_dataset = CRIME_Dataset(out_patch_size   =  out_patch_size, 
                                    phase            =  'valid',
                                    subtract_mean    =  True, 
                                    sub_dataset      =  sub_dataset) 
      return train_dataset, valid_dataset
    
  def data_Transform(self,op_list):
      cur_list = []
      ops = {'vflip':VFlip(),'hflip':HFlip(),'rot90':Rot90()}
      for op_str in op_list:
        cur_list.append(ops[op_str])
      print ('op_list  = {}'.format(cur_list))
      return random_transform(* cur_list)
  
  def label_Generator(self):
      return labelGenerator()

  def optimizer(self,model):
      return optim.Adagrad(filter(lambda x: x.requires_grad, model.parameters()),
                                           lr=self.train_conf['learning_rate'], 
                                           lr_decay=0, 
                                           weight_decay=0)
  @property
  def name(self):
        nstr=self.network.name + '_' \
           + self.train_dataset.name + '_' \
           + '-'.join(self.label_conf['labels']) +'_' \
           + self.data_transform.name
        if 'sub_net' in self.conf:
          nstr = nstr+'_'+'freeze_net1={}'.format(self.conf['sub_net']['freeze_weight'])
        return nstr


class experiment():
  def __init__(self,experiment_config):
      self.exp_cfg           = experiment_config
      
      self.model_saved_dir   = self.exp_cfg.net_conf['model_save_dir']
      self.model_save_steps  = self.exp_cfg.net_conf['model_save_step']
      self.model             = self.exp_cfg.network.float()
      
      self.use_gpu           = self.exp_cfg.net_conf['use_gpu'] and torch.cuda.is_available()
      if self.use_gpu:
            print ('model_set_cuda')
            self.model = self.model.cuda()
      self.use_parallel = False
      
      

      #pre_trained_iter =  self.exp_cfg.net_conf['load_train_iter']
      # if pre_trained_iter > 0:
      #   self.net_load_weights(pre_trained_iter)

      if 'sub_net' in self.exp_cfg.conf and 'trained_file' in self.exp_cfg.conf['sub_net']:
        pre_trained_file = self.exp_cfg.conf['sub_net']['trained_file']
        print('load weights for subnet  from {}'.format(pre_trained_file))
        print ('file exists = {}'.format(os.path.exists(pre_trained_file)))
        self.exp_cfg.sub_network.load_state_dict(torch.load(pre_trained_file))

      if 'trained_file' in self.exp_cfg.net_conf:
        pre_trained_file = self.exp_cfg.net_conf['trained_file']
        print('load weights from {}'.format(pre_trained_file))
        self.model.load_state_dict(torch.load(pre_trained_file))

      if not os.path.exists(self.model_saved_dir):
        os.mkdir(self.model_saved_dir)

      self.mse_loss   = torch.nn.MSELoss()
      self.bce_loss   = torch.nn.BCELoss() 
      self.optimizer  = self.exp_cfg.optimizer(self.model)

  def train(self):
      # set model to the train mode
      boardwriter = tensorBoardWriter()
      self.model.train()
      self.set_parallel_model()
      train_loader = DataLoader(dataset     = self.exp_cfg.train_dataset,
                                batch_size  = self.exp_cfg.net_conf['batch_size'],
                                shuffle     = True,
                                num_workers = 4)
      
      def show_iter_info(iters,runing_loss, iter_str, time_elaps, end_of_iter = False):
        if end_of_iter:
           loss_str    = 'loss : {:.2f}'.format(runing_loss/float(self.model_save_steps))
           printProgressBar(self.model_save_steps, self.model_save_steps, prefix = iter_str, suffix = loss_str, length = 50)
           
        else:
           loss_str = 'loss : {:.2f}'.format(merged_loss.data[0])
           #print ('show merged_loss = {}'.format(merged_loss.data))
           loss_str =  loss_str + ', time : {:.2}s'.format(time_elaps)
           printProgressBar(iters, self.model_save_steps, prefix = iter_str, suffix = loss_str, length = 50)
      
      def get_iter_info(iters):
          iter_range  = (iters+1) // self.model_save_steps
          steps       = (iters+1) % self.model_save_steps
          start_iters = iter_range*self.model_save_steps
          end_iters   = start_iters + self.model_save_steps
          iter_str    = 'iters : {} to {}:'.format(start_iters,end_iters)

          return steps,iter_str

      for epoch in range(5):
            runing_loss = 0.0
            start_time  = time.time()
            train_losses_acumulator = losses_acumulator()

            for i, (data,targets) in enumerate(train_loader, 0):
              data   = Variable(data).float()
              target = self.make_variable(targets)
              if self.use_gpu:
                  data      = data.cuda().float()
                  targets   = self.make_cuda_data(targets)
                
              self.optimizer.zero_grad()
              #print ('data shape ={}'.format(data.data[0].shape))
              preds        = self.model(data)
              losses       = self.compute_loss(preds,targets)

              merged_loss = losses['merged_loss']
              
              merged_loss.backward()
              self.optimizer.step()
              
              runing_loss += merged_loss.data[0]
              time_elaps   = time.time() - start_time
              train_losses_acumulator.append_losses(losses)

              steps,iter_str=get_iter_info(i)

              if steps == 0:
                  self.save_model(i)
                  show_iter_info(steps,runing_loss, iter_str, time_elaps, end_of_iter = True)
                  start_time   = time.time()
                  runing_loss  = 0.0
                  
                  train_losses = train_losses_acumulator.get_ave_losses()
                  valid_losses, (data,preds, targets)=self.valid()
                  boardwriter.write(i,train_losses,valid_losses,data,preds,targets)
                  train_losses_acumulator.reset()
  
              else:
                  show_iter_info(steps,runing_loss, iter_str, time_elaps, end_of_iter = False)
                  # loss_str = 'loss : {:.5f}'.format(general_loss.data[0])
                  # loss_str =  loss_str + ', time : {:.2}s'.format(elaps_time)
                  # printProgressBar(steps, self.model_save_steps, prefix = iters, suffix = loss_str, length = 50)
  
  def valid(self):
        #from torchvision.utils import save_image
        valid_losses_acumulator = losses_acumulator()
        dataset = self.exp_cfg.valid_dataset
        self.model.eval()
        valid_loader = DataLoader(dataset =dataset,
                                  batch_size=1,
                                  shuffle  =True,
                                  num_workers=1)
        loss = 0.0
        iters = 120
        for i, (data,targets) in enumerate(valid_loader, 0):
            #print data.shape
            data    = Variable(data).float()
            targets = self.make_variable(targets)
            if self.use_gpu:
                data     = data.cuda().float()
                targets  = self.make_cuda_data(targets)
            
            preds  = self.model(data)
            losses = self.compute_loss(preds,targets)
            loss += losses['merged_loss'].data[0]
            valid_losses_acumulator.append_losses(losses)
            # loss += self.mse_loss(dist_pred,distance).data[0]
            # label_conf['labels']=label_conf.get('labels',['gradient','sizemap','affinity','centermap','distance'])
            if i % iters ==0:
                exp_config_name = self.exp_cfg.name
                #save2figure(i,'raw_img_'   + exp_config_name, data)
                saveRawfigure(i,'raw_img_' + exp_config_name, data)
                
                if 'distance' in preds:
                  save2figure(i,'dist_t_map_'+ exp_config_name,targets['distance'],)
                  save2figure(i,'dist_p_map_'+ exp_config_name,preds['distance'])
                
                if 'sizemap' in preds:
                  save2figure(i,'size_p_img_' + exp_config_name, torch.log(preds['sizemap']),use_pyplot=True)
                  save2figure(i,'size_t_img_' + exp_config_name, torch.log(targets['sizemap']),use_pyplot=True)

                if 'affinity' in preds:
                  save2figure(i,'affin_t_img_' + exp_config_name, targets['affinity'])
                  save2figure(i,'affin_p_img_' + exp_config_name, preds['affinity'])

                if 'gradient' in preds:
                  ang_t_map=compute_angular(targets['gradient'])
                  ang_p_map=compute_angular(preds['gradient'])
                  save2figure(i,'ang_t_img_' + exp_config_name, ang_t_map,use_pyplot =True)
                  save2figure(i,'ang_p_img_' + exp_config_name, ang_p_map,use_pyplot =True)

                if 'centermap' in preds:
                  save2figure(i,'cent_t_img_x_' + exp_config_name, targets['centermap'][:,0,:,:])
                  save2figure(i,'cent_p_img_x_' + exp_config_name, preds['centermap'][:,0,:,:])
                  save2figure(i,'cent_t_img_y_' + exp_config_name, targets['centermap'][:,1,:,:])
                  save2figure(i,'cent_p_img_y_' + exp_config_name, preds['centermap'][:,1,:,:])

                if 'final' in preds:
                  save2figure(i,'final_dist_t_map_'+ exp_config_name,targets['distance'],)
                  save2figure(i,'final_dist_p_map_'+ exp_config_name,preds['final'])

                #if 'final' in preds:


                #save_image([data.data.cpu(), preds['distance'].data.cpu()], \
                #           exp_config_name +' _valid.png')
                # save_image([data, \
                #            targets['distance'],preds['distance'], \
                #            targets['sizemap'], preds['sizemap'],\
                #            targets['affinity'],preds['affinity'], \
                #            ang_t_map,ang_p_map ], \
                #            exp_config_name +' _valid.png')
                
            if i >= iters-1:
                break
        loss = loss / iters
        self.model.train()
        print (' valid loss : {:.2f}'.format(loss))
        return valid_losses_acumulator.get_ave_losses(), (data,preds, targets)
  
  #def predict(self):
  #  pass


  def net_load_weight(self, iters):
      self.model_file = self.model_saved_dir + '/' \
                         + '{}_iter_{}.model'.format(
                                                     self.experiment_config.name,
                                                     pre_trained_iter)
      print('Load weights  from {}'.format(self.model_file))
      self.model.load_state_dict(torch.load(self.model_file))
    
  def make_variable(self,label_dict):
      for key,value in label_dict.iteritems():
        label_dict[key] = Variable(value).float()
      return label_dict
    
  def make_cuda_data(self,label_dict):
      for key,value in label_dict.iteritems():
        label_dict[key] = value.cuda().float()
      return label_dict
    
  def set_parallel_model(self):
      gpus = [0]
      use_parallel = True if len(gpus) >1 else False
      if use_parallel:
          self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
    
  def compute_loss(self,preds,targets):
        def compute_loss_foreach_label(preds,targets):
            outputs ={}
            #print 'key = {}'.format(preds.keys())
            if 'gradient' in preds:
              ang_loss  = angularLoss(preds['gradient'], targets['gradient'])
              pred_size = np.prod(preds['gradient'].data.shape)
              outputs['ang_loss'] =ang_loss / float(pred_size)

            ''' We want the location of boundary(affinity) in distance map  to be zeros '''
            if 'distance' in preds:
              #print ('distance in  preds')
              distance  = targets['distance'] * (1-targets['affinity'])
              dist_loss = boundary_sensitive_loss(preds['distance'],distance, targets['affinity'])
              pred_size = np.prod(preds['distance'].data.shape)
              outputs['dist_loss']   = dist_loss / float(pred_size)

            # 'labels',['gradient','sizemap','affinity','centermap','distance']
            #if 'affinity' in self.exp_cfg.label_conf:
            if 'affinity' in preds:
              affin_loss=self.bce_loss(torch.sigmoid(preds['affinity']),targets['affinity'])
              pred_size = np.prod(preds['affinity'].data.shape)
              outputs['affinty_loss'] = affin_loss / float(pred_size)

            if 'sizemap' in preds:
              size_loss = self.mse_loss(preds['sizemap'],targets['sizemap'])
              pred_size = np.prod(preds['sizemap'].data.shape)
              outputs['size_loss'] =size_loss /float(pred_size)


            if 'centermap' in preds:
              center_loss = self.mse_loss(preds['centermap'],targets['centermap'])
              pred_size = np.prod(preds['centermap'].data.shape)
              outputs['center_loss'] =center_loss / float(pred_size)
            return outputs

        outputs = {}
        if not self.exp_cfg.train_conf['final_loss_only'] or not 'final' in preds:
          outputs = compute_loss_foreach_label(preds,targets)
            
        if 'final' in preds:
          m_preds ={}
          final_lb = self.exp_cfg.label_conf['final_label']
          m_preds[final_lb] = preds['final']
          fin_loss = compute_loss_foreach_label(m_preds,targets)
          outputs['final_loss']   = fin_loss[fin_loss.keys()[0]]
        
        #print  outputs.keys()

        loss = sum(outputs.values())

        # distance  = targets['distance'] * (1-targets['affinity'])
        # fin_loss = boundary_sensitive_loss(preds['final'], distance, targets['affinity'])
        # outputs['final_dist_loss']   = fin_loss
        ''' As the final output is distance map, we mainly put weights on distance loss''' 
        #loss    = dist_loss + ang_loss + size_loss + center_loss
        # total_loss = 0
        # for loss in outputs:
        #   tota

        #loss      = 0.9995*dist_loss+0.0005*ang_loss
        
        #if train:
        #  loss.backward()

        outputs['merged_loss'] = loss
        return outputs
  def save_model(self,iters):
        model_save_file = self.get_model_save_filename(iters)
        torch.save(self.model.state_dict(),model_save_file)
        print(' saved {}'.format(model_save_file))


  def get_model_save_filename(self,iters):
        model_save_file = self.model_saved_dir + '/' \
                      + '{}_iter_{}.model'.format(self.exp_cfg.name,iters)
        return model_save_file
  
  def predict(self):
        self.model.eval()
        #model.load_state_dict(torch.load(self.model_file))
        # dataset = CRIME_Dataset(out_size  = self.input_size, phase ='valid')
        dataset = self.exp_cfg.valid_dataset
        train_loader = DataLoader(dataset =dataset,
                                  batch_size=1,
                                  shuffle  =True,
                                  num_workers=1)
        for i , (data,target)in enumerate(train_loader,start =0):
            #data, target = Variable(data).float(), Variable(target).float()
            distance = target['distance']
            data, distance = Variable(data).float(), Variable(distance).float()
            if self.use_gpu:
                data   = data.cuda().float()
                distance = distance.cuda().float()

            preds = self.model(data)
            dist_pred = preds['final']
            #loss = self.mse_loss(dist_pred, distance)
            #print('loss:{}'.format(loss.data[0]))
            model_name=self.model.name
            save2figure(i,'dist_p_map_'+self.exp_cfg.name+'_predict',dist_pred,use_pyplot=True)
            save2figure(i,'dist_t_map_'+self.exp_cfg.name+'_predict',distance,use_pyplot=True)
            watershed_d(i,dist_pred)
            if i > 7:
                break


class losses_acumulator():
  def __init__(self):
    self.reset()
  def append_losses(self,current_iter_loss):
    for key, value in current_iter_loss.iteritems():
        if key not in self.total_loss_dict:
          self.total_loss_dict[key] = value.data
        else:
          self.total_loss_dict[key] += value.data
    self.append_iters += 1
  
  def get_ave_losses(self):
    ave_dict ={}
    for key,value in self.total_loss_dict.iteritems():
        ave_dict[key] = value / float(self.append_iters)
    return ave_dict
  
  def reset(self):
    self.total_loss_dict = {}
    self.append_iters = 0



class tensorBoardWriter():
  def __init__(self):
    self.writer = SummaryWriter()
  def write(self,iters,train_loss_dict,valid_loss_dict,data,preds,targets):
    for key, value in train_loss_dict.iteritems():
      self.writer.add_scalar('train_loss/{}'.format(key),value,iters)

    for key,value in valid_loss_dict.iteritems():
      self.writer.add_scalar('valid_loss/{}'.format(key),value,iters)

    for key,value in preds.iteritems():
      if key == 'gradient':
        im = compute_angular(value)
      else:
        im =value.data[0]
      if isinstance(im,Variable):
        im = im.data
      #print('tensorb im shape {} = {}'.format(key, im.shape))
      if key == 'centermap':
        print(im.shape)
        im = torch.unsqueeze(im,1)
        #im = im.permute(1,0,2,3)
        #im = vutils.make_grid(im, normalize=True, scale_each=True)
        #self.writer.add_image('pred/{}'.format(key +'_x'), im, iters)
      #else:
      im = vutils.make_grid(im, normalize=True, scale_each=True)
      #im = matplotlib.cm.spring(im.cpu().numpy())
      #im = torch.FloatTensor(im[:,:,:3])
      self.writer.add_image('pred/{}'.format(key), im, iters)
      # im = vutils.make_grid(im, normalize=True, scale_each=True)
      # self. writer.add_image('predict/{}'.format(key), im, iters)

    for key,value in targets.iteritems():
      if key == 'gradient':
        im = compute_angular(value)
      else:
        im =value
      if isinstance(im,Variable):
         = im.data
      #print('tensorb im shape {} = {}'.format(key, im.shape))
      if key == 'centermap':
        im = im.permute(1,0,2,3)
    
      im2 = im.cpu().numpy()
      im2 = np.squeeze(im2)
      if im2.ndim == 2:
       im2 = np.expand_dims(im2, 0)
      
      im_list = []
      for i in range(im2.shape[0]):
        #cm_d  = matplotlib.cm.cm.gist_earth(im2[i])
        denom = im2[i] - np.min(im2[i])
        im = (denom/max(np.max(denom),0.0000001))
        cm_d  = matplotlib.cm.gist_earth(im)
        #print cm_d.shape
        cm_d = cm_d[:,:,0:3]
        im_list.append(np.transpose(cm_d,(2,0,1)))
      im=np.stack(im_list, axis = 0)
        
      im = torch.FloatTensor(im)
      #if im.dim > 3:
      im = vutils.make_grid(im, normalize=True, scale_each=True)
      #print 'im shape is for {}  = {}'.format(key, im.shape)
     
      #print 'im shape after matplot {}'.format(im.shape)
      self.writer.add_image('target/{}'.format(key), im, iters)
    
    if isinstance(data,Variable):
        data =data.data
    z_dim = data.shape[1]
    for i in range(max(1,z_dim -3+1)):
      img = data[:,i:i+3,:,:]
      raw_im = vutils.make_grid(img, normalize=True, scale_each=True)
      self.writer.add_image('raw_{}'.format(i),raw_im, iters)




def saveRawfigure(iters, file_prefix,output):
    if isinstance(output,Variable):
           output = output.data
    data = output.cpu()
    z_dim = data.shape[1]
    for i in range(max(1,z_dim -3+1)):
      img = data[:,i:i+3,:,:]
      save_image(img,file_prefix+'{}_iter_{}_slice.png'.format(iters,i),normalize =True)

def save2figure(iters,file_prefix,output, use_pyplot =False):
    from torchvision.utils import save_image
    if not use_pyplot:
      if isinstance(output,Variable):
           output = output.data
      data = output.cpu()
      save_image(data,file_prefix+'{}.png'.format(iters),normalize =True)
    else:
      my_dpi = 96
      plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
      if isinstance(output,Variable):
          output = output.data
      data = output.cpu().numpy()
      if data.ndim ==4:
           I = data[0,0]
      elif data.ndim==3:
           I = data[0]
      else:
           I = data
      plt.imshow(I)
      plt.savefig(file_prefix+'{}.png'.format(iters))
      plt.close()

def compute_angular(x):
    # input x must be a 4D data [n,c,h,w]
    if isinstance(x,Variable):
        x = x.data
    x = F.normalize(x)*0.99999
    #print(x.shape)
    x_aix = x[:,0,:,:]/torch.sqrt(torch.sum(x**2,1))
    angle_map   = torch.acos(x_aix)
    return angle_map

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    from torchvision.utils import make_grid
    from PIL import Image
    #tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
def watershed_d(i,distance):
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from skimage.color import label2rgb
    from skimage.morphology import disk,skeletonize
    import skimage
   # from skimage.morphology.skeletonize
    from skimage.filters import gaussian

    if isinstance(distance,Variable):
        distance = distance.data
    my_dpi = 96
    plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
    distance = distance.cpu().numpy()
    distance =np.squeeze(distance)

    hat = ndimage.black_tophat(distance, 14)
    # Combine with denoised image
    hat -= 0.3 * distance
    # Morphological dilation to try to remove some holes in hat image
    hat = skimage.morphology.dilation(hat)


    #local_maxi = peak_local_max(distance, footprint=np.ones((3, 3)),indices=False)
    #from skimage.filters.rank import mean_bilateral
    markers = distance > 3.5
    markers = skimage.morphology.label(markers)
    #distance = mean_bilateral(distance.astype(np.uint16), disk(20), s0=10, s1=10) 
    #distance = gaussian((distance-np.mean(distance))/np.max(np.abs(distance)))   
    #local_maxi = peak_local_max(distance, indices=False, min_distance=5)
   # markers = skimage.morphology.label(local_maxi)[0]
    #markers = ndimage.label(local_maxi, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance, markers)
    



    #ccImage = (distance > 4)
    #labels = skimage.morphology.label(ccImage)
    #labels = skimage.morphology.remove_small_objects(labels, min_size=4)
    #labels = skimage.morphology.remove_small_holes(labels)
    plt.imshow(label2rgb(labels), interpolation='nearest')
    #plt.imshow(labels)
    plt.savefig('seg_{}.png'.format(i))
    plt.imshow(labels, cmap=plt.cm.spectral)
    #plt.imshow(labels)
    plt.savefig('seg_{}_no.png'.format(i))

    plt.imshow(markers)
    plt.savefig('marker_{}.png'.format(i))
    plt.close('all')