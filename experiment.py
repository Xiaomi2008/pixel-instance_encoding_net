from torch_networks.networks import Unet,DUnet, MdecoderUnet
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch_networks.networks import Unet,DUnet,MdecoderUnet 
# from transform import *

from utils.EMDataset import CRIME_Dataset,labelGenerator
from utils.transform import VFlip, HFlip, Rot90, random_transform
from utils.torch_loss_functions import *
from utils.printProgressBar import printProgressBar
import torchvision.utils as vutils
from matplotlib import pyplot as plt
import pytoml as toml
import torch.optim as optim
import time
import pdb


class experiment_config():
  def __init__(self, config_file):
    self.parse_toml(config_file)
    networks = \
              {'Unet' : Unet,'DUnet' : DUnet,'MDUnet': MdecoderUnet}
    network_model           = networks[self.net_conf['model']]
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
    print network_model
    self.network = network_model(target_label = label_ch_pair) 

  def parse_toml(self,file):
      with open(file, 'rb') as fi:
            conf = toml.load(fi)
            net_conf  = conf['network']
            net_conf['model']            = net_conf.get('model','DUnet')
            net_conf['model_saved_dir']  = net_conf.get('model_saved_dir','model')
            net_conf['load_train_iter']  = net_conf.get('load_train_iter', None)
            net_conf['model_save_steps'] = net_conf.get('model_save_steps',500)
            net_conf['patch_size']       = net_conf.get('patch_size',[320,320,1])
           

            label_conf =conf['target_labels']

            #print label_conf

            label_conf['labels']=label_conf.get('labels',['gradient','sizemap','affinity','centermap','distance'])

            data_aug_conf = conf['data_augmentation']

            #print data_aug_conf
            data_aug_conf['transform'] = data_aug_conf.get('transform',['vflip','hflip','rot90'])

            self.label_conf        = label_conf
            self.data_aug_conf     = data_aug_conf
            self.net_conf          = net_conf
            self.dataset_conf      = conf['dataset']



  def dataset(self,out_patch_size,transform):
      sub_dataset = self.dataset_conf['sub_dataset']
      out_patch_size = self.net_conf['patch_size']
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
      op_list = []
      ops = {'vflip':VFlip(),'hflip':HFlip(),'rot90':Rot90()}
      for op_str in op_list:
        op_list.appent(ops[op_str])
      return random_transform(* op_list)
  
  def label_Generator(self):
      return labelGenerator()

  def optimizer(self,model):
      return optim.Adagrad(filter(lambda x: x.requires_grad, model.parameters()),
                                           lr=0.01, 
                                           lr_decay=0, 
                                           weight_decay=0)
  @property
  def name(self):
      return self.network.name + '_' \
           + self.train_dataset.name + '_' \
           + self.data_transform.name


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
      
      

      pre_trained_iter =  self.exp_cfg.net_conf['load_train_iter']
      if pre_trained_iter > 0:
        self.net_load_weights(pre_trained_iter)

      if not os.path.exists(self.model_saved_dir):
        os.mkdir(self.model_saved_dir)

      self.mse_loss   = torch.nn.MSELoss()
      self.optimizer  = self.exp_cfg.optimizer(self.model)

  def train(self):
      # set model to the train mode
      self.model.train()
      self.set_parallel_model()
      train_loader = DataLoader(dataset     = self.exp_cfg.train_dataset,
                                batch_size  = self.exp_cfg.net_conf['batch_size'],
                                shuffle     = True,
                                num_workers = 4)
      
      def show_iter_info(iters,runing_loss, iter_str, time_elaps, end_of_iter = False):
        if end_of_iter:
           loss_str    = 'loss : {:.5f}'.format(runing_loss/float(self.model_save_steps))
           printProgressBar(self.model_save_steps, self.model_save_steps, prefix = iter_str, suffix = loss_str, length = 50)
           self.valid_dist() 
        else:
           loss_str = 'loss : {:.5f}'.format(merged_loss.data[0])
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
            for i, (data,targets) in enumerate(train_loader, 0):
              data   = Variable(data).float()
              target = self.make_variable(targets)
              if self.use_gpu:
                  data      = data.cuda().float()
                  targets   = self.make_cuda_data(targets)
                
              
              preds        = self.model(data)
              losses       = self.compute_loss(preds,targets)
              merged_loss = losses['merged_loss']
              self.optimizer.zero_grad()
              self.optimizer.step()
              
              runing_loss += merged_loss.data[0]
              time_elaps   = time.time() - start_time

              steps,iter_str=get_iter_info(i)

              if steps == 0:
                  self.save_model(i)
                  show_iter_info(steps,runing_loss, iter_str, time_elaps, end_of_iter = True)
                  start_time = time.time()
                  runing_loss = 0.0
              else:
                  show_iter_info(steps,runing_loss, iter_str, time_elaps, end_of_iter = False)
                  # loss_str = 'loss : {:.5f}'.format(general_loss.data[0])
                  # loss_str =  loss_str + ', time : {:.2}s'.format(elaps_time)
                  # printProgressBar(steps, self.model_save_steps, prefix = iters, suffix = loss_str, length = 50)
  
  def valid(self):
        dataset = self.exp_cfg.valid_dataset,
        self.model.eval()
        valid_loader = DataLoader(dataset =dataset,
                                  batch_size=1,
                                  shuffle  =True,
                                  num_workers=1)
        loss = 0.0
        iters = 120
        for i, (data,target) in enumerate(valid_loader, 0):
            target = self.make_variable(targets)
            if self.use_gpu:
                data     = data.cuda().float()
                targets  = self.make_cuda_data(targets)
            preds  = self.model(data)
            losses = self.compute_loss(dist_pred,distance)
            loss += losses['distance'].data[0]
            # loss += self.mse_loss(dist_pred,distance).data[0]
            if i % iters ==0:
                exp_config_name = self.exp_cfg.name()
                save2figuer(i,'dist_t_map_'+ exp_config_name,targets['distance'])
                save2figuer(i,'dist_p_map_'+ exp_config_name,preds['distance'])
                save2figuer(i,'dist_raw_img_' + exp_config_name, data)
            if i >= iters-1:
                break
        loss = loss / iters
        self.model.train()
        print (' valid loss : {:.3f}'.format(loss))
  
  def predict(self):
    pass
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
    
  def compute_loss(self,preds,targets, train = True):
        outputs ={}
        ang_loss  = angularLoss(preds['gradient'], targets['gradient'])

        ''' We want the location of boundary(affinity) in distance map  to be zeros '''
        distance  = targets['distance'] * (1-targets['affinity'])

        dist_loss = boundary_sensitive_loss(preds['distance'],distance, targets['affinity'])

        ''' As the final output is distance map, we mainly put weights on distance loss''' 
        loss      = 0.9995*dist_loss+0.0005*ang_loss
        
        if train:
          loss.backward()

        outputs['dist_loss']   = dist_loss
        outputs['ang_loss']    = ang_loss
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


def save2figuer(iters,file_prefix,output):
    from torchvision.utils import save_image
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
    plt.close('all')