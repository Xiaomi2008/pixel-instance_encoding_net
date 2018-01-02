# from utils.transform import VFlip, HFlip, Rot90, random_transform
import torch
from experiment import experiment, experiment_config
from utils.instance_mask_dataloader \
    import CRIME_Dataset_3D_labels, instance_mask_GTproc_DataLoader, instance_mask_NNproc_DataLoader
from torch_networks.networks import Unet, DUnet, MdecoderUnet, Mdecoder2Unet, \
    MdecoderUnet_withDilatConv, Mdecoder2Unet_withDilatConv, MaskMdecoderUnet_withDilatConv
from torch_networks.gcn import GCN
from utils.torch_loss_functions import *
from utils.printProgressBar import printProgressBar
from torch.autograd import Variable
import time
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np
import pdb
# from matplotlib import pyplot as plt
import matplotlib
from matplotlib import pyplot as plt
import pytoml as toml
import torch.optim as optim
import os


# from utils.torch_loss_functions import dice_loss


class masknet_experiment_config(experiment_config):
    def __init__(self, config_file):
        # super(masknet_experiment_config, self).__init__(config_file)
        self.parse_toml(config_file)

        self.aviable_networks_dict = \
            {'Unet': Unet, 'DUnet': DUnet, 'MDUnet': MdecoderUnet,
             'MDUnetDilat': MdecoderUnet_withDilatConv, \
             'MaskMDnetDilat': MaskMdecoderUnet_withDilatConv,
             'Mdecoder2Unet_withDilatConv': Mdecoder2Unet_withDilatConv,
             'GCN': GCN}

        self.data_transform = self.data_Transform(self.data_aug_conf['transform'])
        # self.label_generator = self.label_Generator()

        self.train_dataset, self.valid_dataset \
            = self.dataset(self.net_conf['patch_size'], self.data_transform)


        ''' total output channel is consisted of z patch( =3), 1 object mask channel, and total channels of predicted label'''
        in_ch = self.net_conf['patch_size'][2] +1 + self.masker_out_chs
        first_out_ch = self.net_conf['first_out_ch']

        #print('mask_out_chs = {}',self.masker_out_chs)

        print (self.net_conf['model'])
        net_model = self.aviable_networks_dict[self.net_conf['model']]

        self.network = net_model(target_label={'mask': 3}, in_ch=in_ch,BatchNorm_final=False,first_out_ch=first_out_ch)

    @property
    def masker_out_chs(self):
        out_lable_dict_from_dataset = self.train_dataset.output_labels()
        ch_count = 0
        for lb in self.masker_conf['labels_cat_in']:
            if lb == 'final':
                lb = 'distance'
            ch_count += out_lable_dict_from_dataset[lb]
        return ch_count

    def get_mask_dataloader(self, phase='train'):
        if phase == 'train':
            batch_size = self.net_conf['batch_size']
            dataset = self.train_dataset
        else:
            batch_size = 1
            dataset = self.valid_dataset
        return self.__mask_loader__(conf_mask=self.masker_conf,
                                    dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=1,
                                    use_gpu=self.net_conf['use_gpu'])

    def __mask_loader__(self, conf_mask, dataset, batch_size, num_workers=1, use_gpu=True):
        mode = conf_mask['mode']

        #print(conf_mask)
        assert (mode == 'GT' or mode == 'NN')

        #print ('mode = {}'.format(mode))
        if mode == 'GT':
            data_loader = instance_mask_GTproc_DataLoader(label_cat_in=conf_mask['labels_cat_in'],
                                                          dataset=dataset,
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          num_workers=num_workers)
        elif mode == 'NN':
            data_out_labels = dataset.output_labels()
            nn_model = self.aviable_networks_dict[conf_mask['nn_model']]
            in_ch = self.net_conf['patch_size'][2]
            nn_model = nn_model(target_label=data_out_labels, in_ch=in_ch).float()
            pre_trained_weights = conf_mask['nn_weight_file']
            # pre_trained_weights = \
            #     '../model/Mdecoder2Unet_withDilatConv_in_3_chs_Dataset-CRIME-All_affinity-sizemap-centermap-distance' \
            #     '-gradient_VFlip-HFlip-Rot90_freeze_net1=True_iter_32499.model'
            nn_model.load_state_dict(torch.load(pre_trained_weights))
            data_loader = instance_mask_NNproc_DataLoader(label_cat_in=conf_mask['labels_cat_in'],
                                                          nn_model=nn_model,
                                                          use_gpu=use_gpu,
                                                          dataset=dataset,
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          num_workers=num_workers
                                                          )
        return data_loader

    def parse_toml(self, file):
        with open(file, 'rb') as fi:
            conf = toml.load(fi)
            print (conf)
            masker_conf = conf['mask_loader']

            masker_conf['mode'] = masker_conf.get('mode', 'NN')
            masker_conf['nn_model'] = masker_conf.get('nn_model', 'MDUnetDilat')
            masker_conf['use_gpu'] = masker_conf.get('use_gpu', True)
            # masker_conf['nn_weight_file'] = masker_conf.get('nn_weight_file', 'None')

            net_conf = conf['network']
            net_conf['model'] = net_conf.get('model', 'DUnet')
            net_conf['first_out_ch']= net_conf.get('first_out_ch',32)
            net_conf['model_saved_dir'] = net_conf.get('model_saved_dir', 'model')
            net_conf['load_train_iter'] = net_conf.get('load_train_iter', None)
            net_conf['model_save_steps'] = net_conf.get('model_save_steps', 500)
            net_conf['patch_size'] = net_conf.get('patch_size', [320, 320, 3])
            # net_conf['trained_file'] = masker_conf.get('trained_file', 'None')

            # net_conf['learning_rate']    = net_conf.get('learning_rate',0.01)

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
        out_patch_size = self.net_conf['patch_size']
        print 'this out {}'.format(out_patch_size)
        train_dataset = CRIME_Dataset_3D_labels(out_patch_size=out_patch_size,
                                                phase='train',
                                                subtract_mean=True,
                                                transform=self.data_transform,
                                                sub_dataset=sub_dataset)

        valid_dataset = CRIME_Dataset_3D_labels(out_patch_size=out_patch_size,
                                                phase='valid',
                                                subtract_mean=True,
                                                sub_dataset=sub_dataset)
        return train_dataset, valid_dataset

    @property
    def name(self):
        nstr = self.masker_conf['mode'] + '_' \
               + self.network.name + '_' \
               + self.train_dataset.name + '_' \
               + 'mask_' \
               + '{}_loss_'.format(self.train_conf['loss_fn']) \
               + self.data_transform.name
        return nstr


class masknet_experiment():
    def __init__(self, masknet_experiment_config):
        self.exp_cfg = masknet_experiment_config
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
        self.bce_loss = StableBCELoss()
        self.bce_logit_loss=torch.nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.mask_bce_loss = StableBalancedMaskedBCE
        #self.bce_loss = torch.nn.BCELoss()

    def train(self):
        # set model to the train mode
        tensorboard_writer = tensorBoardWriter(self.exp_cfg.train_conf['tensorboard_folder'])
        self.model.train()

        train_loader = self.exp_cfg.get_mask_dataloader('train')

        def show_iter_info(iters, runing_loss, iter_str, time_elaps, end_of_iter=False):
            if end_of_iter:
                loss_str = 'loss : {:.2f}'.format(runing_loss / float(self.model_save_steps))
                printProgressBar(self.model_save_steps, self.model_save_steps, prefix=iter_str, suffix=loss_str,
                                 length=50)

            else:
                loss_str = 'loss : {:.2f}'.format(loss.data[0])
                # print ('show merged_loss = {}'.format(merged_loss.data))
                loss_str = loss_str + ', time : {:.2}s'.format(time_elaps)
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
            start_time = time.time()
            train_losses_acumulator = losses_acumulator()

            for i, (data, targets) in enumerate(train_loader, 0):
                # print (targets['distance'].shape)
                data = Variable(data).float()
                targets = dict(map(lambda (k, v): (k, Variable(v)), targets.iteritems()))
                # target = Variable(targets).flaot()
                if self.use_gpu:
                    data = data.cuda().float()
                    targets = dict(map(lambda (k, v): (k, v.cuda().float()), targets.iteritems()))
                    # targets = targets.cuda().float()

                self.optimizer.zero_grad()
                # print ('data shape ={}'.format(data.data[0].shape))
                preds = self.model(data)
                losses = self.compute_loss(preds, targets)
                loss = losses['mask_loss']
                loss.backward()
                self.optimizer.step()
                runing_loss += loss.data[0]
                time_elaps = time.time() - start_time
                train_losses_acumulator.append_losses(losses)

                steps, iter_str = get_iter_info(i)

                if steps == 0:
                    self.save_model(i)
                    show_iter_info(steps, runing_loss, iter_str, time_elaps, end_of_iter=True)
                    start_time = time.time()
                    runing_loss = 0.0

                    train_losses = train_losses_acumulator.get_ave_losses()
                    valid_losses, (data, preds, targets) = self.valid()
                    tensorboard_writer.write(i, train_losses, valid_losses, data, preds, targets)
                    train_losses_acumulator.reset()
                else:
                    show_iter_info(steps, runing_loss, iter_str, time_elaps, end_of_iter=False)
                    # loss_str = 'loss : {:.5f}'.format(general_loss.data[0])
                    # loss_str =  loss_str + ', time : {:.2}s'.format(elaps_time)
                    # printProgressBar(steps, self.model_save_steps, prefix = iters, suffix = loss_str, length = 50)

    def valid(self):
        # from torchvision.utils import save_image
        valid_losses_acumulator = losses_acumulator()
        valid_loader = self.exp_cfg.get_mask_dataloader('valid')
        loss = 0.0
        iters = self.exp_cfg.net_conf['valid_iters']
        self.model.eval()
        for i, (data, targets) in enumerate(valid_loader, 0):
            # print data.shape
            data = Variable(data).float()
            targets = dict(map(lambda (k, v): (k, Variable(v)), targets.iteritems()))
            # targets = self.make_variable(targets)
            if self.use_gpu:
                data = data.cuda().float()
                targets = dict(map(lambda (k, v): (k, v.cuda().float()), targets.iteritems()))
                # targets = self.make_cuda_data(targets)

            preds = self.model(data)
            losses = self.compute_loss(preds, targets)
            loss += losses['mask_loss'].data[0]
            valid_losses_acumulator.append_losses(losses)
            # loss += self.mse_loss(dist_pred,distance).data[0]
            # label_conf['labels']=label_conf.get('labels',['gradient','sizemap','affinity','centermap','distance'])

            if i >= iters - 1:
                break
        loss = loss / iters
        self.model.train()
        print (' valid loss : {:.2f}'.format(loss))
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
        def compute_loss_foreach_label(preds, targets):
            outputs = {}
            #print(preds.keys())
            loss_func_dict ={'dice':self.dice_loss,'bce': self.bce_loss, 
                            'mse': self.mse_loss,'bce_logit': self.bce_logit_loss,
                            'mask_bce': self.mask_bce_loss}

            my_loss=loss_func_dict[self.exp_cfg.train_conf['loss_fn']]
            if 'mask' in preds:
                
                #d_loss = dice_loss(preds['mask'], targets['mask'])
                d_loss = my_loss(preds['mask'], targets['mask'])
                #d_loss = self.bce_loss(preds['mask'], targets['mask'])
                #pred_size = np.prod(preds['mask'].data.shape)
                outputs['mask_loss'] = d_loss 
                #/ float(pred_size)

                #outputs['mask_loss'] = d_loss
                # pred_size = np.prod(preds['mask'].data.shape)
                # outputs['mask_loss'] = dice_loss / float(pred_size)
            return outputs

        outputs = compute_loss_foreach_label(preds, targets)
        loss = sum(outputs.values())
        outputs['merged_loss'] = loss
        # print(outputs.keys())
        return outputs


class losses_acumulator():
    def __init__(self):
        self.reset()

    def append_losses(self, current_iter_loss):
        for key, value in current_iter_loss.iteritems():
            if key not in self.total_loss_dict:
                self.total_loss_dict[key] = value.data
            else:
                self.total_loss_dict[key] += value.data
        self.append_iters += 1

    def get_ave_losses(self):
        ave_dict = {}
        for key, value in self.total_loss_dict.iteritems():
            ave_dict[key] = value / float(self.append_iters)
        return ave_dict

    def reset(self):
        self.total_loss_dict = {}
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



        # self.write_images(preds, 'pred', iters)
        # self.write_images(targets, 'targets', iters)

        # if isinstance(data, Variable):
        #     data = data.data
        # z_dim = data.shape[1]



        # raw_im_list = []
        # for i in range(z_dim):
        #     raw_im_list.append(data[:, i, :, :])
        # raw_images = torch.stack(raw_im_list, dim=0)
        # raw_im = vutils.make_grid(raw_images, normalize=True, scale_each=True)
        # #print('raw_im shape = {}'.format(raw_im.shape))
        # self.writer.add_image('raw_{}'.format(i), raw_im, iters)

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
