from experiment import experiment_config
from utils.EMDataset import slice_dataset
from utils.utils import watershed_seg2D
from utils.evaluation import adapted_rand, voi
import pytoml as toml
import torch
from torch_networks.networks import Unet, DUnet, MdecoderUnet, Mdecoder2Unet, \
    MdecoderUnet_withDilatConv, Mdecoder2Unet_withDilatConv
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb
import pdb
class predict_config():
    def __init__(self, config_file):
        self.parse_toml(config_file)
        NETWORKS = \
            {'Unet': Unet, 'DUnet': DUnet, 'MDUnet': MdecoderUnet,
             'MDUnetDilat': MdecoderUnet_withDilatConv, 'M2DUnet': Mdecoder2Unet,
             'M2DUnet_withDilatConv': Mdecoder2Unet_withDilatConv}
        self.dataset = slice_dataset(sub_dataset=self.dataset_conf['sub_dataset'],
                                subtract_mean=True,
                                split='valid',
                                slices = self.net_conf['z_slices'],
                                data_config='conf/cremi_datasets_with_tflabels.toml')
        data_out_labels = self.dataset.output_labels()
        input_lbCHs_cat_for_net2 = self.label_conf['label_catin_net2']
        
        # create network and load the weights
        net_model = NETWORKS[self.net_conf['model']]
        if self.net_conf['model'] == 'M2DUnet_withDilatConv':
            self.network = net_model(freeze_net1=True,
                                        target_label=data_out_labels, 
                                          label_catin_net2=input_lbCHs_cat_for_net2, 
                                          in_ch=self.net_conf['z_slices'])
            print(net_model)
        else:
            self.network = net_model(target_label=data_out_labels,in_ch=self.net_conf['z_slices'])
        #self.network = net_model(target_label=data_out_labels, in_ch=self.net_conf['z_slices'])
        pre_trained_file = self.net_conf['trained_file']
        print('load weights from {}'.format(pre_trained_file))
        self.network.load_state_dict(torch.load(pre_trained_file))

    def parse_toml(self, file):
        with open(file, 'rb') as fi:
            conf = toml.load(fi)
            net_conf = conf['network']
            net_conf['model'] = net_conf.get('model', 'DUnet')
            net_conf['z_slices'] = net_conf.get('z_slices',3)
 
            label_conf = conf['target_labels']
            label_conf['labels'] = label_conf.get('labels',
                                                  ['gradient', 'sizemap', 'affinity', 'centermap', 'distance'])
            label_conf['final_label'] = label_conf.get('final_labels', 'distance')
       
            self.label_conf = label_conf
            self.net_conf = net_conf
            self.dataset_conf = conf['dataset']
            self.conf = conf


class em_seg_predict():
    def __init__(self,predict_config, seg3D_connector =None):
        self.exp_cfg = predict_config
        self.use_gpu = self.exp_cfg.net_conf['use_gpu'] \
                       and torch.cuda.is_available()
        self.model = self.exp_cfg.network
        self.dataset = self.exp_cfg.dataset
        self.seg3D_connector = seg3D_connector


    def predict(self):
        pred_seg_2d = self.__predict2D__()
        pred_seg_3d = self.__make_3Dseg__(self.dataset.get_data(),pred_seg_2d.cpu().numpy())
        return pred_seg_3d



    def __predict2D__(self):
        if self.use_gpu:
            print ('model_set_cuda')
            self.model = self.model.cuda().float()
        self.model.eval()
        pred_seg=torch.zeros_like(torch.from_numpy(self.dataset.get_data().astype(int)))
        pred_seg=pred_seg.long()
        p_shape = pred_seg.shape
        cut_size = 1248
        print('pred_seg shape {}'.format(pred_seg.shape))
        pred_seg=pred_seg[:p_shape[0],:cut_size,:cut_size]
        raw_Data =self.dataset.get_data()
        #g_seg    =self.dataset.get_label()
        print('dataset len = {}'.format(len(self.dataset)))
        for i in range(len(self.dataset)):
            out = self.dataset.__getitem__(i)
            out_data    = out['data'][:,:,:cut_size,:cut_size]
            #print(out['label'])
            g_seg_data  =out['label'][:,:,:cut_size,:cut_size]
            #print('out_data shape = {}'.format(out_data) )

            data = Variable(out_data,volatile=True).float()
            print('data input shape {}'.format(data.data.shape))
            if self.use_gpu:
               data = data.cuda().float()
            preds = self.model(data)
            watershed_d = np.squeeze(watershed_seg2D(preds['distance']))

            # gradient_d = np.squeeze(preds['gradient'].data.cpu().numpy())
            # print('w shape = {}'.format(watershed_d.shape))
            # print('g shape = {}'.format(gradient_d.shape))
            # #plt.imshow(watershed_d.astype(np.int))
            # plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
            my_dpi = 96
            # fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
            
            # a = fig.add_subplot(3, 1, 1)
            # d =np.squeeze(out_data.cpu().numpy())
            
            # print('d shape = {}'.format(d.shape))
            
            # imgplot = plt.imshow(d[1])
            # a.set_title('im')
            
            # a = fig.add_subplot(3, 1, 2)
            # plt.imshow(label2rgb(watershed_d.astype(np.int)), interpolation='nearest')
            # a.set_title('p_seg')

            #seg_d =np.squeeze(g_seg_data.cpu().numpy())
            g_seg_in = np.squeeze(g_seg_data.cpu().numpy())[1]
            #print('seg_d shape = {}'.format(seg_d.shape))
            # a = fig.add_subplot(3, 1, 3)
            # plt.imshow(label2rgb(g_seg_in), interpolation='nearest')
            # a.set_title('g_seg')
            
            #plt.savefig('watershed_d_{}.png'.format(i))
            #plt.close()
            #centermap = np.squeeze(preds['centermap'].data.cpu().numpy())

            arand_eval=adapted_rand(watershed_d.astype(np.int),g_seg_in)
            print('arand = {} '.format(arand_eval))
            #voi_d =voi(watershed_d.astype(np.int),g_seg_in)
            #print('arand = {} voi ={}'.format(arand_eval,void_d))

            pred_seg[i]= torch.from_numpy(watershed_d)
        return pred_seg

    def __make_3Dseg__(self, data, pred_Seg2D):

        if self.seg3D_connector:
            return self.seg3D_connector(data, pred_Seg2D)
        seg_connector =Simple_MaxCoverage_3DSegConnector()
        seg3d = seg_connector(data,pred_Seg2D)
        return seg3d


class Simple_MaxCoverage_3DSegConnector(object):
    # def __init__(self,data,seg2d):
    #   self.data = data
    #   self.seg2d =seg2d
    def __call__(self,data,seg2d):
        '''first, we need make sure that there are no same ids between slices''' 
        seg2d = self.reset_slice_id(seg2d)
        seg3d=self.update_sliceS_seg(seg2d,order ='down')
        #seg3d=self.update_sliceS_seg(seg3d,order ='up')
        return seg3d
    def reset_slice_id(self,seg2d):
        for i in range(len(seg2d)):
            seg2d[i]+=(1300*i)
        return seg2d
    def update_sliceS_seg(self,seg2d,order ='down'):
        seg3d = seg2d.copy()
        slice_idxs = range(len(seg2d)) if order == 'down' else range(len(seg2d))[::-1]
        for i in range(len(slice_idxs)-1):
            ref_idx = slice_idxs[i]
            update_idx=slice_idxs[i+1]
            
            seg_slice_1 = seg3d[ref_idx].copy()
            seg_slice_2 = seg3d[update_idx].copy()
            unique_ids_1,count_1 = np.unique(seg_slice_1,return_counts = True)
            unique_ids_2,count_2 = np.unique(seg_slice_2,return_counts = True)
            s1_id_size = dict(zip(unique_ids_1,count_1))
            s2_id_size = dict(zip(unique_ids_2,count_2))
            idx = np.argsort(count_1)
            '''first connect to small objects, and then larger ones,
               This will ensure that larger object can overide to make final connections
               .which is crutial to IOU based measurement'''
            #sort_uids =unique_ids_1[idx]
            sort_uids =unique_ids_1[idx][::-1]
            '''-------------------------------------------------------------------------'''
            connected_ids ={sid:False for sid in unique_ids_2}
            for uid in sort_uids:
                bool_mask = (seg_slice_1==uid)
                mask_ids = seg_slice_2[bool_mask]
                uids_2, count = np.unique(mask_ids, return_counts=True)
                idx = np.argmax(count)
                max_cover_id = uids_2[idx]
                #max_cover_size = np.sum( seg3d[update_idx][seg3d[update_idx]==max_cover_id].astype(int))

                #refer_id_size = np.sum(bool_mask.astype(np.int))
                max_cover_size = s2_id_size[max_cover_id]
                refer_id_size =  s1_id_size[uid]
                # if abs(0.5 - float(max_cover_size)/float((max_cover_size+refer_id_size))) < 0.2:
                #     seg3d[update_idx][seg3d[update_idx]==max_cover_id] =uid
                if connected_ids[max_cover_id]:

                    #if order =='up':
                        #seg3d[seg3d==max_cover_id]=uid
                    seg3d[seg3d==uid]=max_cover_id
                        #print('connected_ids occurs')
                else:
                    seg3d[update_idx][seg3d[update_idx]==max_cover_id] =uid
                connected_ids[uid]=True
        return seg3d

                


class em_seg_eval(object):
    def __init__(self,predict_config):
        self.exp_cfg =predict_config
        self.seg_predictor  = em_seg_predict(self.exp_cfg)

    def eval(self):
        seg_3d ={}
        arand_eval  ={}
        voi_eval ={}

        print('subset data = {}'.format(self.exp_cfg.dataset.subset))
        for d_set in self.exp_cfg.dataset.subset:
            #d_set = 'Set_A'
            print('predicting {} ...'.format(d_set))
            self.exp_cfg.dataset.set_current_subDataset(d_set)
            seg_lbs = self.exp_cfg.dataset.get_label()
            seg_3d[d_set]=self.seg_predictor.predict()
            x_size = seg_3d[d_set].shape[1]
            y_size = seg_3d[d_set].shape[2]
            seg_lbs= seg_lbs[:,:x_size,:y_size]
            arand_eval[d_set]=adapted_rand(seg_3d[d_set],seg_lbs)

            my_dpi = 96
            fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)

            label2d_seg = label2rgb(seg_3d[d_set])
            
            a = fig.add_subplot(1, 2, 1)
            #plt.imshow(label2rgb(np.squeeze(seg_3d[d_set][1])))
            plt.imshow(label2d_seg [0], interpolation='nearest')
            a.set_title('upper_seg')
            
            a = fig.add_subplot(1, 2, 2)
            plt.imshow(label2d_seg [1], interpolation='nearest')
            a.set_title('lower_seg')

            plt.show()


            print('arand for {} = {}'.format(d_set,arand_eval[d_set]))
            #voi_eval[d_set] = voi(seg_3d[d_set],seg_lbs)
            void_eval = 0
        return arand_eval, voi_eval