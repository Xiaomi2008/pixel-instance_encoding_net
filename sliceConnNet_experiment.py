# from utils.transform import VFlip, HFlip, Rot90, random_transform
import torch
from torch.utils.data import DataLoader
from experiment import experiment, experiment_config
from torch.autograd import Variable
from utils.mask_slice_pair_dataset import CRIME_Dataset_3D_mask_pair

from torch_networks.slice_connection_wideResnet import Wide_ResNet
from torch_networks.resnext import resnext50
from utils.torch_loss_functions import *
from utils.printProgressBar import printProgressBar

import time
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt
import pytoml as toml
import torch.optim as optim
import os


class slice_connect_experiment_config(experiment_config):
    def __init__(self, config_file):
        # super(masknet_experiment_config, self).__init__(config_file)
        self.parse_toml(config_file)

        NETWORKS = \
            {'Wide_ResNet': Wide_ResNet,
            'resnet50': resnext50}

        self.data_transform = self.data_Transform(self.data_aug_conf['transform'])
        # self.label_generator = self.label_Generator()

        self.train_dataset, self.valid_dataset \
            = self.dataset(self.net_conf['in_patch_size'], self.data_transform)

        in_ch = 4
        print (self.net_conf['model'])
        net_model = NETWORKS[self.net_conf['model']]

        #self.network = net_model(target_label={'mask': 3}, in_ch=in_ch,BatchNorm_final=False,first_out_ch=first_out_ch)
        if self.net_conf['model'] =='Wide_ResNet':
            self.network = net_model(10, 4, 0.3, 2)
        elif self.net_conf['model'] =='resnet50':
            #self.network = net_model(8, 5, 2)
            self.network = net_model(8, 5, 2)
    def parse_toml(self, file):
        with open(file, 'rb') as fi:
            conf = toml.load(fi)
            print (conf)
            masker_conf = conf['mask_loader']
            masker_conf['mode'] = masker_conf.get('mode', 'NN')
            masker_conf['use_gpu'] = masker_conf.get('use_gpu', True)
            masker_conf['nn_weight_file'] = masker_conf.get('nn_weight_file', 'None')

            net_conf = conf['network']
            net_conf['model'] = net_conf.get('model', 'Wide_ResNet')
            net_conf['model_saved_dir'] = net_conf.get('model_saved_dir', 'model')
            net_conf['model_save_steps'] = net_conf.get('model_save_steps', 500)

            train_conf = conf['train']
            train_conf['learning_rate'] = train_conf.get('learning_rate', 0.01)
            train_conf['tensorboard_folder'] = train_conf.get('tensorboard_folder', 'runs/exp1')
            

            data_aug_conf = conf['data_augmentation']
            data_aug_conf['transform'] = data_aug_conf.get('transform', ['vflip', 'hflip', 'rot90'])

            self.data_aug_conf = data_aug_conf
            self.masker_conf = masker_conf
            self.net_conf = net_conf
            self.dataset_conf = conf['dataset']
            self.train_conf = train_conf
            self.conf = conf




    def dataset(self, out_patch_size, transform):
        sub_dataset = self.dataset_conf['sub_dataset']
        out_patch_size = self.net_conf['in_patch_size']
        print 'this out {}'.format(out_patch_size)

        train_dataset = CRIME_Dataset_3D_mask_pair(
                                         phase='train',
                                         subtract_mean=True,
                                         in_patch_size=(600,600,2),
                                         out_patch_size=(224, 224, 2),
                                         sub_dataset=sub_dataset,
                                         transform=self.data_transform)
        valid_dataset = CRIME_Dataset_3D_mask_pair(
                                         phase='valid',
                                         subtract_mean=True,
                                         in_patch_size=(600,600,2),
                                         out_patch_size=(224, 224, 2),
                                         sub_dataset=sub_dataset,
                                         transform=None)
        return train_dataset, valid_dataset


    def get_mask_dataloader(self, phase='train'):
        if phase == 'train':
            batch_size = self.net_conf['batch_size']
            dataset = self.train_dataset
            num_workers =8
        else:
            batch_size = 70
            dataset = self.valid_dataset
            num_workers = 1
        return self.__mask_loader__(conf_mask=self.masker_conf,
                                    dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    use_gpu=self.net_conf['use_gpu'])

    def __mask_loader__(self, conf_mask, dataset, batch_size, num_workers=1, use_gpu=True):
        mode = conf_mask['mode']

        #print(conf_mask)
        assert (mode == 'GT' or mode == 'NN')

        #print ('mode = {}'.format(mode))
        if mode == 'GT':
            # data_loader = instance_mask_GTproc_DataLoader(label_cat_in=conf_mask['labels_cat_in'],
            #                                               dataset=dataset,
            #                                               batch_size=batch_size,
            #                                               shuffle=True,
            #                                               num_workers=num_workers)


            data_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
        elif mode == 'NN':
            # data_out_labels = dataset.output_labels()
            # nn_model = self.aviable_networks_dict[conf_mask['nn_model']]
            # in_ch = self.net_conf['patch_size'][2]
            # nn_model = nn_model(target_label=data_out_labels, in_ch=in_ch).float()
            # pre_trained_weights = conf_mask['nn_weight_file']
            # nn_model.load_state_dict(torch.load(pre_trained_weights))
            # data_loader = instance_mask_NNproc_DataLoader(label_cat_in=conf_mask['labels_cat_in'],
            #                                               nn_model=nn_model,
            #                                               use_gpu=use_gpu,
            #                                               dataset=dataset,
            #                                               batch_size=batch_size,
            #                                               shuffle=True,
            #                                               num_workers=num_workers
            #                                               )
            pass
        return data_loader



    @property
    def name(self):
        nstr = self.masker_conf['mode'] + '_' \
               + self.network.name + '_' \
               + self.train_dataset.name + '_' \
               + 'mask_' \
               + '{}_loss_'.format(self.train_conf['loss_fn']) \
               + self.data_transform.name
        return nstr


class slice_connect_experiment():
    def __init__(self, slice_connect_experiment_config):
        self.exp_cfg = slice_connect_experiment_config
        self.model_saved_dir = self.exp_cfg.net_conf['model_save_dir']
        self.model_save_steps = self.exp_cfg.net_conf['model_save_step']
        self.model = self.exp_cfg.network.float()

        self.use_gpu = self.exp_cfg.net_conf['use_gpu'] and torch.cuda.is_available()
        if self.use_gpu:
            print ('model_set_cuda')
            self.model = self.model.cuda()
        self.use_parallel = False

        if 'trained_file' in self.exp_cfg.net_conf:
            pre_trained_file = self.exp_cfg.net_conf['trained_file']
            print('load weights from {}'.format(pre_trained_file))
            self.model.load_state_dict(torch.load(pre_trained_file))

        if not os.path.exists(self.model_saved_dir):
            os.mkdir(self.model_saved_dir)

       
        # self.bce_loss = torch.nn.BCELoss()
        # self.dice_loss =
        self.optimizer = self.exp_cfg.optimizer(self.model)
        self.criterion =nn.CrossEntropyLoss()
        #self.bce_loss = StableBCELoss()
        #self.bce_logit_loss=torch.nn.BCEWithLogitsLoss()
        #self.dice_loss = DiceLoss()
        #self.mse_loss = torch.nn.MSELoss()
        #self.mask_bce_loss = StableBalancedMaskedBCE
        #self.bce_loss = torch.nn.BCELoss()

    def train(self):
        # set model to the train mode
        tensorboard_writer = tensorBoardWriter(self.exp_cfg.train_conf['tensorboard_folder'])
        self.model.train()

        train_loader = self.exp_cfg.get_mask_dataloader('train')

        def show_iter_info(iters, runing_loss, acc, iter_str, time_elaps, end_of_iter=False):
            if end_of_iter:
                loss_str = 'loss : {:.2f}, acc : {:.2f}'.format(float(runing_loss) / float(self.model_save_steps),acc)
                printProgressBar(self.model_save_steps, self.model_save_steps, prefix=iter_str, suffix=loss_str,
                                 length=50)

            else:
                loss_str = 'loss : {:.2f}, acc: {:.2f} '.format(loss.data[0],acc)
                loss_str = loss_str + ', time : {:.2f}'.format(time_elaps)
                printProgressBar(iters, self.model_save_steps, prefix=iter_str, suffix=loss_str, length=50)

        def get_iter_info(iters):
            iter_range = (iters + 1) // self.model_save_steps
            steps = (iters + 1) % self.model_save_steps
            start_iters = iter_range * self.model_save_steps
            end_iters = start_iters + self.model_save_steps
            iter_str = 'iters : {} to {}:'.format(start_iters, end_iters)

            return steps, iter_str

        for epoch in range(5):
            runing_loss = 0.0
            total =0
            correct =0
            start_time = time.time()
            train_losses_acumulator = losses_acumulator()

            for i, (data, targets) in enumerate(train_loader, 0):
               data = Variable(data).float()
               targets = Variable(targets).long()
               if self.use_gpu:
                data = data.cuda().float()
                targets = targets.cuda().long()

                self.optimizer.zero_grad()
                # print ('data shape ={}'.format(data.data[0].shape))
                preds = self.model(data)
                loss = self.compute_loss(preds, targets)
                loss.backward()
                self.optimizer.step()
                runing_loss += loss.data
                time_elaps = time.time() - start_time
                train_losses_acumulator.append_losses(loss)
                steps, iter_str = get_iter_info(i)
                _, predicted = torch.max(preds.data, 1)
                total = targets.size(0)
                correct = predicted.eq(targets.data).cpu().sum()
                acc = float(correct) / float(total)

                if steps == 0:
                    self.save_model(i)
                    show_iter_info(steps, runing_loss, acc, iter_str, time_elaps, end_of_iter=True)
                    start_time = time.time()
                    runing_loss = 0.0

                    train_losses = train_losses_acumulator.get_ave_losses()
                    valid_losses, (data, preds, targets) = self.valid()
                    #tensorboard_writer.write(i, train_losses, valid_losses, data, preds, targets)
                    train_losses_acumulator.reset()
                    total=0
                    correct=0
                else:
                    show_iter_info(steps, runing_loss, acc, iter_str, time_elaps, end_of_iter=False)

    def valid(self):
        # from torchvision.utils import save_image
        valid_losses_acumulator = losses_acumulator()
        valid_loader = self.exp_cfg.get_mask_dataloader('valid')
        loss = 0.0
        iters = self.exp_cfg.net_conf['valid_iters']
        self.model.eval()
        total =0 
        correct =0 
        for i, (data, targets) in enumerate(valid_loader, 0):
            data = Variable(data,volatile=True).float()
            targets = Variable(targets,volatile=True).long()
            if self.use_gpu:
                data = data.cuda().float()
                targets = targets.cuda().long()
                # targets = self.make_cuda_data(targets)

            preds = self.model(data)
            loss+= self.compute_loss(preds, targets)
            valid_losses_acumulator.append_losses(loss)
            _, predicted = torch.max(preds.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            # loss += self.mse_loss(dist_pred,distance).data[0]
            # label_conf['labels']=label_conf.get('labels',['gradient','sizemap','affinity','centermap','distance'])

            if i >= iters - 1:
                break

        acc = float(correct) / float(total)
        loss = float(loss) / float(iters)
        self.model.train()
        print (' valid loss : {:.2f},  acc : {:.2f}'.format(loss,acc))
        return valid_losses_acumulator.get_ave_losses(), (data, preds, targets)

    def save_model(self, iters):
        model_save_file = self.get_model_save_filename(iters)
        torch.save(self.model.state_dict(), model_save_file)
        print(' saved {}'.format(model_save_file))

    def get_model_save_filename(self, iters):
        model_save_file = self.model_saved_dir + '/' \
                          + '{}_iter_{}.model'.format(self.exp_cfg.name, iters)
        return model_save_file

    def compute_loss(self, preds, targets):
        loss_func_dict ={'crossEntropy':self.criterion}

        my_loss=loss_func_dict[self.exp_cfg.train_conf['loss_fn']]
        loss = my_loss(preds, targets)
        return loss

class losses_acumulator():
    def __init__(self):
        self.reset()

    def append_losses(self, current_iter_loss):
        self.append_iters += 1
        self.total_loss+=current_iter_loss

    def get_ave_losses(self):
        ave=self.total_loss/ float(self.append_iters)
        return ave

    def reset(self):
        self.total_loss = 0.0
        self.append_iters = 0


class tensorBoardWriter():
    def __init__(self, save_folder=None):
        if not save_folder:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(save_folder)

    def wirte_model_graph(self, model, lastvar):
        self.writer.add_graph(model, lastvar)

    def write(self, iters, train_losses, valid_losses, data, preds, targets):
        for key, value in train_losses.iteritems():
            self.writer.add_scalar('train_loss/{}'.format(key), value, iters)

        for key, value in valid_losses.iteritems():
            self.writer.add_scalar('valid_loss/{}'.format(key), value, iters)


        pred_mask = preds['mask']
        self.write_ch_slice_images(pred_mask,'preds',iters)

        targ_mask = targets['mask']
        self.write_ch_slice_images(targ_mask,'targets',iters)


        self.write_ch_slice_images(data,'inputs',iters)


    def write_ch_slice_images(self, data, name, iters):
        if isinstance(data, Variable):
            data = data.data
        z_dim = data.shape[1]



        raw_im_list = []
        for i in range(z_dim):
            im = data[:, i, :, :]
            if torch.max(im) - torch.min(im) > 0:
                raw_im_list.append(im)
        raw_images = torch.stack(raw_im_list, dim=0)
        raw_im = vutils.make_grid(raw_images,normalize =True, scale_each=True)
        #raw_im = vutils.make_grid(raw_images, normalize=True, scale_each=True)
        #print('raw_im shape = {}'.format(raw_im.shape))
        self.writer.add_image(name, raw_im, iters)

    def write_images(self, output, name, iters):

        im = output['mask']
        if isinstance(im, Variable):
            im = im.data
        im2 = np.squeeze(im.cpu().numpy())
        if im2.ndim == 2:
            im2 = np.expand_dims(im2, 0)

        im = torch.FloatTensor(np.stack(im2, axis=0))
        im = vutils.make_grid(im, normalize=True, scale_each=True)
        self.writer.add_image(name, im, iters)
