import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch_networks.networks import Unet, DUnet, MdecoderUnet, Mdecoder2Unet, \
    MdecoderUnet_withDilatConv,  MdecoderUnet_withFullDilatConv, Mdecoder2Unet_withDilatConv,\
    Mdecoder2Unet_withDilatConv_LSTM_on_singleOBJ, MdecoderUnet_withDilatConv_centerGate

#from torch_networks.unet3D import MdecoderUnet3D
from torch_networks.res_3D2Dhybrid_unet import \
     hybrid_2d3d_unet, hybrid_2d3d_unet_mutlihead, hybrid_2d3d_unet_mutlihead_with_3section_conv
from utils.EMDataset import CRIME_Dataset, labelGenerator, CRIME_Dataset3D, labelGenerator3D
from utils.transform import VFlip, HFlip, ZFlip, Rot90, NRot90, random_transform, RandomContrast
from utils.torch_loss_functions import *
from utils.printProgressBar import printProgressBar
from utils.utils import watershed_seg
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pytoml as toml
import time
import pdb
import os

class experiment_config():
    def __init__(self, config_file):
        self.parse_toml(config_file)
        print('=====================******===================')

        print self.label_conf['labels']
        print '_'.join(self.label_conf['labels'])
        networks = \
            {'Unet': Unet, 'DUnet': DUnet, 'MDUnet': MdecoderUnet,
             'MDUnetDilat': MdecoderUnet_withDilatConv, 
             'MDUnet_FullDilat':MdecoderUnet_withFullDilatConv,
             'M2DUnet': Mdecoder2Unet,
             'M2DUnet_withDilatConv': Mdecoder2Unet_withDilatConv,
             'M2DUnet_withDilatConv_CLSTM_ObjOut':Mdecoder2Unet_withDilatConv_LSTM_on_singleOBJ,
             'MDUnetDilatCenterGate': MdecoderUnet_withDilatConv_centerGate,
             'MDUnet3D':hybrid_2d3d_unet,
             'MDUnet3D_mhead':hybrid_2d3d_unet_mutlihead,
             'MDUnet3D_sectionConv_mhead': hybrid_2d3d_unet_mutlihead_with_3section_conv}

        self.data_transform = self.data_Transform(self.data_aug_conf['transform'])
       


        self.data_channel_axis = np.argmin(self.net_conf['patch_size'])


        if self.train_conf['final_loss_only'] and 'final_labels' in self.label_conf:
            label_in_use = self.label_conf['final_labels']
        elif 'final_labels' in self.label_conf:
            label_in_use = list(set(self.label_conf['labels'] + self.label_conf['final_labels']))
        elif 'final_label' in self.label_conf:
            label_in_use = [self.label_conf['final_label']]
        else:
            label_in_use = self.label_conf['labels']

        self.label_generator = self.label_Generator(label_in_use)
        self.train_dataset, self.valid_dataset \
            = self.dataset(self.net_conf['patch_size'], 
                           self.data_transform,
                           label_config = label_in_use,
                           channel_axis=self.data_channel_axis, 
                           output_3D=self.dataset_conf['output_3D'])

        # labels_in_use = ['gradient','sizemap','affinity','centermap','distance']


        label_ch_pair_info ={'gradient':2,'sizemap':1,'affinity':1,'centermap':2,'distance':1,'skeleton':1}

        label_ch_pair = {}
        # data_out_labels is a dict that stores "label name" as key and # of channel for that label" as value
        
        data_out_labels = self.train_dataset.output_labels()
        for lb in label_in_use:
            label_ch_pair[lb] = data_out_labels[lb]

        print ('label and ch = {}'.format(label_ch_pair))

        #in_ch = self.net_conf['patch_size'][2]
        
        in_ch = 1 if self.dataset_conf['output_3D'] else self.net_conf['patch_size'][self.data_channel_axis]


        self.sub_network = None
        freeze_net1 = True
        if 'sub_net' in self.conf:
            net_1_ch_pair = {}
            for lb in self.label_conf['labels']:
                net_1_ch_pair[lb] = label_ch_pair_info[lb]
            subnet_model = networks[self.conf['sub_net']['model']]
            self.sub_network = subnet_model(target_label=net_1_ch_pair, in_ch=in_ch)

            #net_model = networks[self.net_conf['model']]
            #self.network = net_model(self.sub_network, freeze_net1=self.conf['sub_net']['freeze_weight'])

        #pdb.set_trace()


        if 'final_labels' in self.label_conf:
            net2_out_put_label=self.label_conf['final_labels']
            net2_target_label_ch_dict= {}
            for lb,ch in data_out_labels.iteritems():
                if lb in net2_out_put_label:
                    net2_target_label_ch_dict[lb] =ch
        elif 'final_label' in self.label_conf:
            net2_target_label_ch_dict= {}
            net2_target_label_ch_dict['final']=data_out_labels[self.label_conf['final_label']]

        net_model = networks[self.net_conf['model']]
        #print(net_model)
        out_ch =1
        print(self.label_conf['final_label'])
        if self.label_conf['final_label'] == 'softmask':
            out_ch =24
        print('out_ch = {}'.format(out_ch))
        print('==================================================')
        #pdb.set_trace()
        if self.net_conf['model'] in ['M2DUnet_withDilatConv','M2DUnet_withDilatConv_CLSTM_ObjOut']:
            input_lbCHs_cat_for_net2 = self.label_conf['label_catin_net2']
            self.network = net_model(self.sub_network, 
                                     freeze_net1=freeze_net1,
                                     target_label=net_1_ch_pair,
                                     net2_target_label= net2_target_label_ch_dict,
                                     label_catin_net2=input_lbCHs_cat_for_net2,
                                     in_ch=in_ch,
                                     out_ch=out_ch,
                                     first_out_ch=16)
        else:
            self.network = net_model(target_label=label_ch_pair, 
                                     in_ch=in_ch, 
                                     BatchNorm_final=False)

    def parse_toml(self, file):
        with open(file, 'rb') as fi:
            conf = toml.load(fi)
            net_conf = conf['network']
            net_conf['model'] = net_conf.get('model', 'DUnet')
            net_conf['model_saved_dir'] = net_conf.get('model_saved_dir', 'model')
            net_conf['load_train_iter'] = net_conf.get('load_train_iter', None)
            net_conf['model_save_steps'] = net_conf.get('model_save_steps', 500)
            net_conf['patch_size'] = net_conf.get('patch_size', [320, 320, 1])
            # net_conf['learning_rate']    = net_conf.get('learning_rate',0.01)

            train_conf = conf['train']
            train_conf['final_loss_only'] = train_conf.get('final_loss_only', False)
            train_conf['learning_rate'] = train_conf.get('learning_rate', 0.01)
            train_conf['tensorboard_folder'] = train_conf.get('tensorboard_folder', 'runs/exp1')
            # net_conf['trained_file']     = net_conf.get('trained_file','')

            label_conf = conf['target_labels']

            label_conf['labels'] = label_conf.get('labels',
                                                  ['gradient', 'sizemap', 'affinity', 'centermap', 'distance'])
            
            label_conf['final_label'] = label_conf.get('final_label', 'distance')
            

            data_aug_conf = conf['data_augmentation']

            # print data_aug_conf
            data_aug_conf['transform'] = data_aug_conf.get('transform', ['vflip', 'hflip', 'rot90','nrot90'])

            self.dataset_conf = conf['dataset']
            self.dataset_conf['output_3D'] = self.dataset_conf.get('output_3D',False)

            self.label_conf = label_conf
            self.data_aug_conf = data_aug_conf
            self.net_conf = net_conf
            
            self.train_conf = train_conf
            self.conf = conf

    def dataset(self, out_patch_size, transform, label_config=None, channel_axis =None, output_3D = False):
        sub_dataset = self.dataset_conf['sub_dataset']
        out_patch_size = self.net_conf['patch_size']
        print 'this out {}'.format(out_patch_size)
        if not channel_axis:
            channel_axis = np.argmin(out_patch_size)

        dataset_class = CRIME_Dataset3D if output_3D else CRIME_Dataset

        train_dataset = dataset_class(out_patch_size=out_patch_size,
                                          phase='train',
                                          subtract_mean=True,
                                          transform=self.data_transform,
                                          sub_dataset=sub_dataset,
                                          channel_axis=channel_axis,
                                          label_config = label_config)

        valid_dataset = dataset_class(out_patch_size=out_patch_size,
                                          phase='valid',
                                          subtract_mean=True,
                                          sub_dataset=sub_dataset,
                                          channel_axis=channel_axis,
                                          label_config = label_config)
        return train_dataset, valid_dataset

    def data_Transform(self, op_list):
        cur_list = []
        ops = {'vflip': VFlip(), 'hflip': HFlip(), 'zflip':ZFlip(), 'rot90': Rot90(),'nrot90':NRot90()}
        for op_str in op_list:
            cur_list.append(ops[op_str])
        #contrast_transform = RandomContrast(0.15,1.5)
        print ('op_list  = {}'.format(cur_list))
        return random_transform(cur_list)

    def label_Generator(self, label_config):
        lb_gen = labelGenerator3D() if self.dataset_conf['output_3D'] else labelGenerator(label_config)
        return lb_gen

    def optimizer(self, model):
        print('op learning_Rate = {}'.format(self.train_conf['learning_rate']))
        model_param = filter(lambda x: x.requires_grad, model.parameters())
        optimizer_dict = {'Adgrad':optim.Adagrad, 'SGD':optim.SGD, 'Adam':optim.Adam}
        lr = self.train_conf['learning_rate']
        if self.train_conf['optimizer'] == 'Adagrad':
            return optim.Adagrad(model_param,
                                     lr=lr,
                                     lr_decay=0,
                                     weight_decay=0)
        elif self.train_conf['optimizer'] =='SGD':
            return optim.SGD(model_param, lr=lr, momentum=0.9)
        elif self.train_conf['optimizer'] =='Adam':
            return optim.Adam(model_param, lr = lr)

    @property
    def name(self):
        #pdb.set_trace()
        nstr = self.network.name + '_' \
               + self.train_dataset.name + '_' \
               + '-'.join(self.label_conf['labels']) + '_' \
               + self.data_transform.name +'_' \
               + 'patch ='+str(self.net_conf['patch_size'])

        if 'sub_net' in self.conf:
            nstr = nstr + '_' + 'freeze_net1={}'.format(self.conf['sub_net']['freeze_weight'])
        return nstr

class experiment():
    def __init__(self, experiment_config):
        self.exp_cfg = experiment_config

        #pdb.set_trace()

        self.model_saved_dir = self.exp_cfg.net_conf['model_save_dir']
        self.model_save_steps = self.exp_cfg.net_conf['model_save_step']
        self.model = self.exp_cfg.network.float()

        self.use_gpu = self.exp_cfg.net_conf['use_gpu'] and torch.cuda.is_available()
        if self.use_gpu:
            print ('model_set_cuda')
            self.model = self.model.cuda()
        self.use_parallel = False


        if 'sub_net' in self.exp_cfg.conf and 'trained_file' in self.exp_cfg.conf['sub_net']:
            pre_trained_file = self.exp_cfg.conf['sub_net']['trained_file']
            print('load weights for subnet  from {}'.format(pre_trained_file))
            print ('file exists = {}'.format(os.path.exists(pre_trained_file)))
            self.exp_cfg.sub_network.load_state_dict(torch.load(pre_trained_file))

        if 'trained_file' in self.exp_cfg.net_conf:
            pre_trained_file = self.exp_cfg.net_conf['trained_file']
            print('load weights from {}'.format(pre_trained_file))

            #if hasattr(self.model, 'set_multi_gpus'):
            #    self.model.set_multi_gpus([0,1])
            self.model.load_state_dict(torch.load(pre_trained_file))

        if not os.path.exists(self.model_saved_dir):
            os.mkdir(self.model_saved_dir)

        self.mse_loss = torch.nn.MSELoss()
        #self.bce_loss = torch.nn.BCELoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.softIOU_match_loss = softIOU_match_loss()
        self.optimizer = self.exp_cfg.optimizer(self.model)

    def train(self):
        # set model to the train mode
        boardwriter = tensorBoardWriter(self.exp_cfg.train_conf['tensorboard_folder'])
        self.model.train()
        #self.set_parallel_model()
        graph_write_done = False
        train_loader = DataLoader(dataset=self.exp_cfg.train_dataset,
                                  batch_size=self.exp_cfg.net_conf['batch_size'],
                                  shuffle=True,
                                  num_workers=4)

        def show_iter_info(iters, runing_loss, iter_str, time_elaps, end_of_iter=False):
            if end_of_iter:
                loss_str = 'loss : {:.2f}'.format(runing_loss / float(self.model_save_steps))
                printProgressBar(self.model_save_steps, self.model_save_steps, prefix=iter_str, suffix=loss_str,
                                 length=50)

            else:
                loss_str = 'loss : {:.2f}'.format(merged_loss.data[0])
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
            train_losses_accumulator = losses_accumulator()

            for i, (data, targets) in enumerate(train_loader, 0):
                #print(data.size())
                data = Variable(data).float()
                target = self.make_variable(targets)
                if self.use_gpu:
                    data = data.cuda().float()
                    targets = self.make_cuda_data(targets)

                self.optimizer.zero_grad()
                # print ('data shape ={}'.format(data.data[0].shape))
                preds = self.model(data)
                losses,t_masks = self.compute_loss(preds, targets)
                # if not graph_write_done:
                #     boardwriter.wirte_model_graph(self.model, preds['gradient'])
                #     graph_write_done = True

                merged_loss = losses['merged_loss']

                merged_loss.backward()
                self.optimizer.step()

                runing_loss += merged_loss.data[0]
                time_elaps = time.time() - start_time
                train_losses_accumulator.append_losses(losses)

                steps, iter_str = get_iter_info(i)

                if steps == 0:
                    self.save_model(i)
                    show_iter_info(steps, runing_loss, iter_str, time_elaps, end_of_iter=True)
                    start_time = time.time()
                    runing_loss = 0.0

                    train_losses = train_losses_accumulator.get_ave_losses()
                    valid_losses, (data, preds, targets) = self.valid()
                    boardwriter.write(i, train_losses, valid_losses, data, preds, targets)
                    train_losses_accumulator.reset()

                else:
                    show_iter_info(steps, runing_loss, iter_str, time_elaps, end_of_iter=False)
                    # loss_str = 'loss : {:.5f}'.format(general_loss.data[0])
                    # loss_str =  loss_str + ', time : {:.2}s'.format(elaps_time)
                    # printProgressBar(steps, self.model_save_steps, prefix = iters, suffix = loss_str, length = 50)

    def valid(self):
        # from torchvision.utils import save_image
        valid_losses_accumulator = losses_accumulator()
        dataset = self.exp_cfg.valid_dataset
        self.model.eval()
        valid_loader = DataLoader(dataset=dataset,
                                  batch_size=5,
                                  shuffle=True,
                                  num_workers=2)
        loss = 0.0
        iters = 35
        for i, (data, targets) in enumerate(valid_loader, 0):
            # print data.shape
            data = Variable(data,volatile=True).float()
            targets = self.make_variable(targets,volatile=True)
            if self.use_gpu:
                data = data.cuda().float()
                targets = self.make_cuda_data(targets)

            preds = self.model(data)
            losses,t_masks = self.compute_loss(preds, targets)

            loss += losses['merged_loss'].data[0]
            valid_losses_accumulator.append_losses(losses)
            if not isinstance(t_masks,int):
                targets['t_masks'] = t_masks
            # loss += self.mse_loss(dist_pred,distance).data[0]
            # label_conf['labels']=label_conf.get('labels',['gradient','sizemap','affinity','centermap','distance'])
            # if i % iters == 0:
            #     exp_config_name = self.exp_cfg.name
            #     # save2figure(i,'raw_img_'   + exp_config_name, data)
            #     saveRawfigure(i, 'raw_img_' + exp_config_name, data)

            #     if 'distance' in preds:
            #         save2figure(i, 'dist_t_map_' + exp_config_name, targets['distance'], )
            #         save2figure(i, 'dist_p_map_' + exp_config_name, preds['distance'])

            #     if 'sizemap' in preds:
            #         save2figure(i, 'size_p_img_' + exp_config_name, torch.log(preds['sizemap']), use_pyplot=True)
            #         save2figure(i, 'size_t_img_' + exp_config_name, torch.log(targets['sizemap']), use_pyplot=True)

            #     if 'affinity' in preds:
            #         save2figure(i, 'affin_t_img_' + exp_config_name, targets['affinity'])
            #         save2figure(i, 'affin_p_img_' + exp_config_name, preds['affinity'])

            #     if 'gradient' in preds:
            #         ang_t_map = compute_angular(targets['gradient'])
            #         ang_p_map = compute_angular(preds['gradient'])
            #         save2figure(i, 'ang_t_img_' + exp_config_name, ang_t_map, use_pyplot=True)
            #         save2figure(i, 'ang_p_img_' + exp_config_name, ang_p_map, use_pyplot=True)

            #     if 'centermap' in preds:
            #         save2figure(i, 'cent_t_img_x_' + exp_config_name, targets['centermap'][:, 0, :, :])
            #         save2figure(i, 'cent_p_img_x_' + exp_config_name, preds['centermap'][:, 0, :, :])
            #         save2figure(i, 'cent_t_img_y_' + exp_config_name, targets['centermap'][:, 1, :, :])
            #         save2figure(i, 'cent_p_img_y_' + exp_config_name, preds['centermap'][:, 1, :, :])

            #     if 'final' in preds:
            #         save2figure(i, 'final_dist_t_map_' + exp_config_name, targets['distance'], )
            #         save2figure(i, 'final_dist_p_map_' + exp_config_name, preds['final'])

            if i >= iters - 1:
                break
        loss = loss / iters
        self.model.train()
        print (' valid loss : {:.2f}'.format(loss))
        return valid_losses_accumulator.get_ave_losses(), (data, preds, targets)

    # def predict(self):
    #  pass

    def net_load_weight(self, iters):
        self.model_file = self.model_saved_dir + '/' \
                          + '{}_iter_{}.model'.format(
                            self.experiment_config.name,
                            pre_trained_iter)
        print('Load weights  from {}'.format(self.model_file))
        self.model.load_state_dict(torch.load(self.model_file))

    def make_variable(self, label_dict, volatile=False):
        for key, value in label_dict.iteritems():
            label_dict[key] = Variable(value, volatile=volatile).float()
        return label_dict

    def make_cuda_data(self, label_dict):
        for key, value in label_dict.iteritems():
            label_dict[key] = value.cuda().float()
        return label_dict

    def set_parallel_model(self):
        gpus = [0,1]
        use_parallel = True if len(gpus) > 1 else False
        if use_parallel:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)

    def compute_loss(self, preds, targets):
        def compute_loss_foreach_label(preds, targets):
            outputs = {}
            # print 'key = {}'.format(preds.keys())
            t_masks = 0
            if 'gradient' in preds:
                ang_loss = angularLoss(preds['gradient'], targets['gradient'])
                pred_size = np.prod(preds['gradient'].data.shape)
                outputs['ang_loss'] = ang_loss / float(pred_size)

            ''' We want the location of boundary(affinity) in distance map  to be zeros '''
            if 'distance' in preds:
                # print ('distance in  preds')
                distance = targets['distance'] * (1 - targets['affinity'])
                # print 'distance = {}'.format(distance.data.shape)
                dist_loss = boundary_sensitive_loss(preds['distance'], distance, targets['affinity'])
                #pred_size = np.prod(preds['distance'].data.shape)
                outputs['dist_loss'] = dist_loss #/ float(pred_size)

            if 'distance2D' in preds:
                target_affinity2D = ((targets['affinityX'] + targets['affinityY'])>0).float()
                target_distance = targets['distance2D'] * (1-target_affinity2D)
                dist_loss = boundary_sensitive_loss(preds['distance2D'],  target_distance, target_affinity2D)
                #pred_size = np.prod(preds['distance2D'].data.shape)
                outputs['dist2D_loss'] = dist_loss #/ float(pred_size)

            if 'distance3D' in preds:
                #target_affinity3D = ((targets['affinityX'] + targets['affinityY'] + targets['affinityZ'])>2).astype(np.int)

                affinity2D = ((targets['affinityX'] + targets['affinityY'])>0).float()

                #print('affinity 2D shape ={}'.format(affinity2D.shape))


                affinity_2D_list =[((affinity2D[:,i] + targets['affinityZ'][:,1])>0).float() for i in range(affinity2D.shape[1])]
                target_affinity3D =  torch.stack(affinity_2D_list,1)


                #print('target af3d shape ={}'.format(target_affinity3D.shape))
                #print('pred af3d shape ={}'.format(targets['distance3D'].shape))
                
                target_distance = targets['distance3D'] * (1-target_affinity3D)
                dist_loss = boundary_sensitive_loss(preds['distance3D'],  target_distance, target_affinity3D)
                #pred_size = np.prod(preds['distance3D'].data.shape)
                outputs['dist3D_loss'] = dist_loss #/ float(pred_size)
            # 'labels',['gradient','sizemap','affinity','centermap','distance']
            # if 'affinity' in self.exp_cfg.label_conf:
            if 'affinity' in preds:
                #affin_loss = self.bce_loss(torch.sigmoid(preds['affinity']), targets['affinity'])
                affin_loss = self.bce_loss(preds['affinity'], targets['affinity'])
                outputs['affinty_loss'] =  affin_loss
                # pred_size = np.prod(preds['affinity'].data.shape)
                # outputs['affinty_loss'] = affin_loss / float(pred_size)

            if 'affinityX' in preds:
                affin_loss = self.bce_loss(preds['affinityX'], targets['affinityX'])
                outputs['affinty_lossX'] =  affin_loss

            if 'affinityY' in preds:
                affin_loss = self.bce_loss(preds['affinityY'], targets['affinityY'])
                outputs['affinty_lossY'] =  affin_loss

            if 'affinityZ' in preds:
                affin_loss = self.bce_loss(preds['affinityZ'], targets['affinityZ'])
                outputs['affinty_lossZ'] =  affin_loss
            
            if 'skeleton' in preds:
                skel_loss = self.bce_loss(preds['skeleton'], targets['skeleton'])
                outputs['skeleton_loss'] =  skel_loss * 10
            
            if 'sizemap' in preds:
                size_loss = self.mse_loss(preds['sizemap'], targets['sizemap'])
                #outputs['size_loss'] = size_loss
                #pred_size = np.prod(preds['sizemap'].data.shape)
                outputs['size_loss'] = size_loss / 200.0

            if 'centermap' in preds:
                center_loss = self.mse_loss(preds['centermap'], targets['centermap'])
                outputs['center_loss'] = center_loss
                #pred_size = np.prod(preds['centermap'].data.shape)
                #outputs['center_loss'] = center_loss / float(pred_size)
            
            if 'softmask' in preds:
                #softmask_loss,t_masks= self.softIOU_match_loss(preds['softmask'],targets['seg'])
                #print ('softmask output shape ={}'.format(preds['softmask'].shape))
                softmask_loss,t_masks= self.softIOU_match_loss(preds['softmask'],targets['seg'])
                outputs['softmask_loss'] = softmask_loss
            return outputs, t_masks

        outputs = {}
        if not self.exp_cfg.train_conf['final_loss_only'] or not 'final' in preds:
            outputs,t_masks = compute_loss_foreach_label(preds, targets)

        if 'final' in preds:
            m_preds = {}
            final_lb = self.exp_cfg.label_conf['final_label']
            m_preds[final_lb] = preds['final']
            fin_loss,t_masks = compute_loss_foreach_label(m_preds, targets)
            outputs['final_loss'] = fin_loss[fin_loss.keys()[0]]


        if 'final_labels' in self.exp_cfg.label_conf and self.exp_cfg.train_conf['final_loss_only']:
            m_preds = {}
            final_lbs = self.exp_cfg.label_conf['final_labels']
            for lb in final_lbs:
                m_preds[lb]=preds[lb]
            
            fin_loss,_ = compute_loss_foreach_label(m_preds, targets)
            outputs.update(fin_loss)




        loss = sum(outputs.values())
        outputs['merged_loss'] = loss
        return outputs,t_masks

    def save_model(self, iters):
        model_save_file = self.get_model_save_filename(iters)
        torch.save(self.model.state_dict(), model_save_file)
        print(' saved {}'.format(model_save_file))

    def get_model_save_filename(self, iters):
        model_save_file = self.model_saved_dir + '/' \
                          + '{}_iter_{}.model'.format(self.exp_cfg.name, iters)
        return model_save_file

    def predict(self):
        self.model.eval()
        # model.load_state_dict(torch.load(self.model_file))
        # dataset = CRIME_Dataset(out_size  = self.input_size, phase ='valid')
        dataset = self.exp_cfg.valid_dataset
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=1)
        for i, (data, target) in enumerate(train_loader, start=0):
            # data, target = Variable(data).float(), Variable(target).float()
            distance = target['distance']
            data, distance = Variable(data).float(), Variable(distance).float()
            if self.use_gpu:
                data = data.cuda().float()
                distance = distance.cuda().float()

            preds = self.model(data)
            dist_pred = preds['final']
            # loss = self.mse_loss(dist_pred, distance)
            # print('loss:{}'.format(loss.data[0]))
            # model_name = self.model.name
            save2figure(i, 'dist_p_map_' + self.exp_cfg.name + '_predict', dist_pred, use_pyplot=True)
            save2figure(i, 'dist_t_map_' + self.exp_cfg.name + '_predict', distance, use_pyplot=True)
            watershed_seg(i, dist_pred)
            if i > 7:
                break


class losses_accumulator():
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


    def write(self, iters, train_loss_dict, valid_loss_dict, data, preds, targets):
        for key, value in train_loss_dict.iteritems():
            self.writer.add_scalar('train_loss/{}'.format(key), value, iters)

        for key, value in valid_loss_dict.iteritems():
            self.writer.add_scalar('valid_loss/{}'.format(key), value, iters)

        self.write_images(preds, 'preds', iters)
        self.write_images(targets, 'targets', iters)

        if isinstance(data, Variable):
            data = data.data
        #print('data dim = {}'.format(data.dim))
        z_dim = data.shape[1]
        raw_im_list = []
        if data.dim() == 4:
            for i in range(max(1, z_dim - 3 + 1)):
                raw_im_list.append(data[:, i:i + 3, :, :])
            raw_images = torch.cat(raw_im_list, dim=0)
        elif data.dim() ==5:
            raw_images = data[0]
            raw_images = raw_images.permute(1,0,2,3)
        raw_images = vutils.make_grid(raw_images, normalize=True, scale_each=True)
        self.writer.add_image('raw_img', raw_images, iters)

    def write_images(self, output_dict, dict_name, iters):
        
        def add_slice_image(x):
            assert x.ndim ==3
            im_list =[]
            for i in range(x.shape[0]):
                denom = x[i] - np.min(x[i])
                im = (denom / max(np.max(denom), 0.0000001))
                cm_d = matplotlib.cm.gist_earth(im)[:, :, 0:3]
                im_list.append(np.transpose(cm_d, (2, 0, 1)))

            return im_list



        for key, value in output_dict.iteritems():
            if key == 'gradient':
                im = compute_angular(value)
            else:
                im = value

            if key in['skeleton','affinity','affinityX','affinityY','affinityZ']:
                im = torch.sigmoid(im)
                im = im if key =='affinity' else 1 -im 

            if isinstance(im, Variable):
                im = im.data
            #print('tensorb key = {}'.format(key))
            #print('tensorb im shape {} = {}'.format(key, im.shape))
            
            '''save only one image'''
            im2 = np.squeeze(im[0].cpu().numpy())
            # if key == 'centermap':
            #     im = im.permute(1, 0, 2, 3)

            if im2.ndim == 2:
                im2 = np.expand_dims(im2, 0)
            
            im_list = []
            
            '''stak over the channel'''
            if im2.ndim == 4:
                # im is 4 D image where each channel is a 3D image
                for ch_im in range(im2.shape[0]):
                    im_list+=add_slice_image(im2[ch_im])
            elif im2.ndim ==3:
                im_list=add_slice_image(im2)
            # for i in range(im2.shape[0]):
            #     denom = im2[i] - np.min(im2[i])
            #     im = (denom / max(np.max(denom), 0.0000001))
            #     cm_d = matplotlib.cm.gist_earth(im)[:, :, 0:3]
            #     im_list.append(np.transpose(cm_d, (2, 0, 1)))

            im = torch.FloatTensor(np.stack(im_list, axis=0))
            im = vutils.make_grid(im, normalize=True, scale_each=True)
            self.writer.add_image('{}/{}'.format(dict_name, key), im, iters)

def saveRawfigure(iters, file_prefix, output):
    if isinstance(output, Variable):
        output = output.data
    data = output.cpu()
    z_dim = data.shape[1]
    for i in range(max(1, z_dim - 3 + 1)):
        img = data[:, i:i + 3, :, :]
        save_image(img, file_prefix + '{}_iter_{}_slice.png'.format(iters, i), normalize=True)


def save2figure(iters, file_prefix, output, use_pyplot=False):
    from torchvision.utils import save_image
    if not use_pyplot:
        if isinstance(output, Variable):
            output = output.data
        data = output.cpu()
        save_image(data, file_prefix + '{}.png'.format(iters), normalize=True)
    else:
        my_dpi = 96
        plt.figure(figsize=(1250 / my_dpi, 1250 / my_dpi), dpi=my_dpi)
        if isinstance(output, Variable):
            output = output.data
        data = output.cpu().numpy()
        if data.ndim == 4:
            I = data[0, 0]
        elif data.ndim == 3:
            I = data[0]
        else:
            I = data
        plt.imshow(I)
        plt.savefig(file_prefix + '{}.png'.format(iters))
        plt.close()


def compute_angular(x):
    # input x must be a 4D data [n,c,h,w]
    if isinstance(x, Variable):
        x = x.data
    x = F.normalize(x) * 0.99999
    # print(x.shape)
    x_aix = x[:, 0, :, :] / torch.sqrt(torch.sum(x ** 2, 1))
    angle_map = torch.acos(x_aix)
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
    # tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


# def watershed_d(i, distance):
#     from scipy import ndimage
#     from skimage.feature import peak_local_max
#     from skimage.segmentation import watershed
#     from skimage.color import label2rgb
#     from skimage.morphology import disk, skeletonize
#     import skimage
#     # from skimage.morphology.skeletonize
#     from skimage.filters import gaussian

#     if isinstance(distance, Variable):
#         distance = distance.data
#     my_dpi = 96
#     plt.figure(figsize=(1250 / my_dpi, 1250 / my_dpi), dpi=my_dpi)
#     distance = distance.cpu().numpy()
#     distance = np.squeeze(distance)

#     hat = ndimage.black_tophat(distance, 14)
#     # Combine with denoised image
#     hat -= 0.3 * distance
#     # Morphological dilation to try to remove some holes in hat image
#     hat = skimage.morphology.dilation(hat)

#     # local_maxi = peak_local_max(distance, footprint=np.ones((3, 3)),indices=False)
#     # from skimage.filters.rank import mean_bilateral
#     markers = distance > 3.5
#     markers = skimage.morphology.label(markers)
#     # distance = mean_bilateral(distance.astype(np.uint16), disk(20), s0=10, s1=10)
#     # distance = gaussian((distance-np.mean(distance))/np.max(np.abs(distance)))
#     # local_maxi = peak_local_max(distance, indices=False, min_distance=5)
#     # markers = skimage.morphology.label(local_maxi)[0]
#     # markers = ndimage.label(local_maxi, structure=np.ones((3, 3)))[0]
#     labels = watershed(-distance, markers)

#     # ccImage = (distance > 4)
#     # labels = skimage.morphology.label(ccImage)
#     # labels = skimage.morphology.remove_small_objects(labels, min_size=4)
#     # labels = skimage.morphology.remove_small_holes(labels)
#     plt.imshow(label2rgb(labels), interpolation='nearest')
#     # plt.imshow(labels)
#     plt.savefig('seg_{}.png'.format(i))
#     plt.imshow(labels, cmap=plt.cm.spectral)
#     # plt.imshow(labels)
#     plt.savefig('seg_{}_no.png'.format(i))

#     plt.imshow(markers)
#     plt.savefig('marker_{}.png'.format(i))
#     plt.close('all')
