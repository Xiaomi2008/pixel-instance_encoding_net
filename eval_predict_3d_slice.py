# from experiment import experiment_config
from utils.EMDataset import slice_dataset
from utils.utils import watershed_seg, watershed_seg2
from utils.evaluation import adapted_rand, voi
import pytoml as toml
import torch
from torch_networks.networks import Unet, DUnet, MdecoderUnet, Mdecoder2Unet, \
    MdecoderUnet_withDilatConv, Mdecoder2Unet_withDilatConv
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb
import h5py
import pdb
class predict_config():
    def __init__(self, config_file):
        self.parse_toml(config_file)
        NETWORKS = \
            {'Unet': Unet, 'DUnet': DUnet, 'MDUnet': MdecoderUnet,
             'MDUnetDilat': MdecoderUnet_withDilatConv, 'M2DUnet': Mdecoder2Unet,
             'M2DUnet_withDilatConv': Mdecoder2Unet_withDilatConv}

        if self.dataset_conf['dataset'] == 'valid':
            data_config = 'conf/cremi_datasets.toml'
            split = 'valid'
        elif self.dataset_conf['dataset'] == 'predict':
            data_config = 'conf/cremi_datasets_test.toml'
            split = 'predict'
        else:
            data_config = 'conf/cremi_datasets.toml'
            split = 'valid'

        def build_network_and_dataset(net_conf, label_conf, slice_axis):
            data_set = slice_dataset(sub_dataset=self.dataset_conf['sub_dataset'],
                                     subtract_mean=True,
                                     split=split,
                                     slices=net_conf['slices'],
                                     slice_axis = slice_axis,
                                     data_config=data_config)

            data_out_labels = data_set.output_labels()
            input_lbCHs_cat_for_net2 = label_conf['label_catin_net2']
            net_model = NETWORKS[net_conf['model']]
            if net_conf['model'] == 'M2DUnet_withDilatConv':
                network = net_model(freeze_net1=True,
                                     target_label=data_out_labels,
                                     label_catin_net2=input_lbCHs_cat_for_net2,
                                     in_ch=net_conf['slices'])
                print(net_model)
            else:
                network = net_model(target_label=data_out_labels, in_ch=net_conf['slices'])


            pre_trained_file = net_conf['trained_file']
            print('load weights from {}'.format(pre_trained_file))
            network.load_state_dict(torch.load(pre_trained_file))
            return network, data_set


        self.front_slice_network, self.front_slice_dataset = \
          build_network_and_dataset(net_conf=self.front_net_conf,label_conf=self.front_label_conf, slice_axis=0)

        self.side_slice_network1, self.side_slice_dataset1 = \
          build_network_and_dataset(net_conf=self.side_net_conf,label_conf=self.side_label_conf, slice_axis=1)

        self.side_slice_network2, self.side_slice_dataset2 = \
          build_network_and_dataset(net_conf=self.side_net_conf,label_conf=self.side_label_conf, slice_axis=2)
    
    def set_current_dataset(self, setname):
        self.front_slice_dataset.set_current_subDataset(setname)
        self.side_slice_dataset1.set_current_subDataset(setname)
        self.side_slice_dataset2.set_current_subDataset(setname)
    
    def get_net_and_dataset(self,axis):
        if axis == 0:
            model   = self.front_slice_network
            dataset = self.front_slice_dataset
            use_gpu = self.front_net_conf['use_gpu'] and torch.cuda.is_available()
        elif axis == 1:
            model   = self.side_slice_network1
            dataset = self.side_slice_dataset1
            use_gpu = self.side_net_conf['use_gpu'] and torch.cuda.is_available()
        elif axis == 2:
            model   = self.side_slice_network2
            dataset = self.side_slice_dataset2
            use_gpu = self.side_net_conf['use_gpu'] and torch.cuda.is_available()
        return model,dataset, use_gpu

    def parse_toml(self, file):
        with open(file, 'rb') as fi:
            conf = toml.load(fi)
            front_net_conf = conf['fontslice_network']
            front_net_conf['model'] = front_net_conf.get('model', 'DUnet')
            front_net_conf['slices'] = front_net_conf.get('slices',3)

            front_label_conf = conf['fronslice_target_labels']
            front_label_conf['labels'] = front_label_conf.get('labels',
                                                  ['gradient', 'sizemap', 'affinity', 'centermap', 'distance'])
            front_label_conf['final_label'] = front_label_conf.get('final_labels', 'distance')


            side_net_conf = conf['sideslice_network']
            side_net_conf['model'] = side_net_conf.get('model', 'DUnet')
            side_net_conf['slices'] = side_net_conf.get('slices',5)

            side_label_conf = conf['sideslice_target_labels']
            side_label_conf['labels'] =side_label_conf.get('labels',
                                                  ['gradient', 'sizemap', 'affinity', 'centermap', 'distance'])
            side_label_conf['final_label'] =side_label_conf.get('final_labels', 'distance')
       
            self.front_net_conf = front_net_conf
            self.front_label_conf = front_label_conf
            self.side_net_conf = side_net_conf
            self.side_label_conf = side_label_conf
            self.dataset_conf = conf['dataset']
            self.conf = conf


class em_seg_predict():
    def __init__(self, predict_config, seg3D_connector=None):
        self.exp_cfg = predict_config
        #self.use_gpu = self.exp_cfg.net_conf['use_gpu'] and torch.cuda.is_available()
        #self.model = self.exp_cfg.network
        #self.dataset = self.exp_cfg.dataset
        #self.seg3D_connector = seg3D_connector

    def predict(self):
       


        #print('pred_dist type is {}'.format(pred_dist_2d_0.type()))
        pred_dist_2d_0 = extend_tTensor_to_1250(self.__predict2D__(axis=0))
        pred_dist_2d_1 = extend_tTensor_to_1250(self.__predict2D__(axis=1))
        pred_dist_2d_2 = extend_tTensor_to_1250(self.__predict2D__(axis=2))
        
        pred_affinty_2d_0 = pred_dist_2d_0 > 1.5
        pred_affinty_2d_1 = pred_dist_2d_1 > 1.5
        pred_affinty_2d_2 = pred_dist_2d_2 > 1.5

        #pred_dist_combine = pred_dist_2d_0 + pred_dist_2d_1 + pred_dist_2d_2
        pred_dist_combine = pred_dist_2d_0 + pred_dist_2d_1 + pred_dist_2d_2
        #pred_dist_combine =  (pred_affinty_2d_0 +  pred_affinty_2d_1 +  pred_affinty_2d_2)
        # show_figure(pred_dist_combine, mode='original', slice_idx = 0)
        # show_figure(pred_dist_combine, mode='original', slice_idx = 10)
        # show_figure(pred_dist_combine, mode='original', slice_idx = 20)
        # show_figure(pred_dist_combine, mode='original', slice_idx = 30)
        # show_figure(pred_dist_combine, mode='original', slice_idx = 40)
        # print('connecting slices ids ...')
        #pred_seg_3dconn= self.__make_3Dseg__(self.dataset.get_data(), pred_seg_2d.cpu().numpy())
        #pred_seg_3dWS  = watershed_seg(pred_dist_combine, threshold=22)

        return pred_dist_2d_0 , pred_dist_combine
        #pred_seg_3dWS  = watershed_seg2(pred_dist_2d_0[:7],pred_dist_combine[:7], threshold=22)
        #pdb.set_trace()
        #return pred_seg_3dWS
        #return pred_seg_3dconn, pred_seg_3dWS
        #return pred_seg_3dconn, pred_seg_3dWS
    
    def __resize_input_for_network__(self,input_data):
        shape = input_data.size()
        if torch.Size([shape[2],shape[3]]) == torch.Size([1250,1250]):
            data = input_data[:,:,:1248,:1248]
        elif torch.Size([shape[2],shape[3]])  == torch.Size([125,1250]):
            data = input_data[:,:,:,:1248]
            complement_data = data[:,:,-1:,:].repeat(1,1,3,1)
            data = torch.cat([data,complement_data],2)
        else:
            raise ValueError('input data shape is not expected {}'.format(shape))
        return data

    def __predict2D__(self, axis=0):
        model,dataset,use_gpu=self.exp_cfg.get_net_and_dataset(axis)
        if use_gpu:
            print ('model_set_cuda')
            model = model.cuda().float()
        model.eval()
        pred_seg = torch.zeros_like(torch.from_numpy(dataset.get_data().astype(int)))
        pred_seg = pred_seg.float()
        p_shape = pred_seg.shape

        pred_dist = torch.zeros_like(torch.from_numpy(dataset.get_data().astype(float)))
        cut_size = 1248
        print('pred_seg shape {}'.format(pred_seg.shape))
        assert axis < 3 and axis >=0
        if axis ==0:
             #pred_seg  = pred_seg[:p_shape[0],1248,:1248]
             pred_dist = pred_dist[:p_shape[0],:1248,:1248]
        elif axis ==1 or axis ==2:
             #pred_seg  = pred_seg[:p_shape[0],128,:1248]
             pred_dist = pred_dist[:p_shape[0],:128,:1248]
        #print('pred_seg shape is {}'.format(pred_seg.shape))
        raw_Data = dataset.get_data()
        #g_seg    =self.dataset.get_label()
        #print('dataset len = {}'.format(len(self.dataset)))
        for i in range(len(dataset)):
            out = dataset.__getitem__(i)
            out_data = self.__resize_input_for_network__(out['data'])
            # out_data = out['data'][:,:,::,:cut_size]
            g_seg_data = None
            if 'label' in out:
                # g_seg_data = out['label'][:,:,::,:cut_size]
                g_seg_data =  self.__resize_input_for_network__(out['label'])
            data = Variable(out_data,volatile=True).float()
            #print('predict slice {}, shape ={}'.format(i,data.data.shape))
            if use_gpu:
                data = data.cuda().float()
            preds = model(data)
            #watershed_d = np.squeeze(watershed_seg((preds['final'] + preds['distance'])/2.0))

            distance_d  = (preds['final'].data +  preds['distance'].data)/2.0

            #distance_d  = preds['final'].data
            #distance_d  = preds['distance'].data
            if g_seg_data is not None:
                g_seg_in = np.squeeze(g_seg_data.cpu().numpy())[1]
                #print('g_seg = shape {}'.format(np.squeeze(g_seg_data.cpu().numpy()).shape))
                #arand_eval = adapted_rand(watershed_d.astype(np.int), g_seg_in)
                #print('arand = {} '.format(arand_eval))
            #pred_seg[i,:,:]= torch.from_numpy(watershed_d)
            #print('pred_shape = {} and distance shape ={}'.format(pred_dist.shape, distance_d.shape))
            print('predicting slice {} ...'.format(i))
            if distance_d.size()[2]==128:
                pred_dist[i,:,:]= torch.squeeze(distance_d[0,0,:125,:])
            else:
                pred_dist[i,:,:]= torch.squeeze(distance_d)
        del model
        
        pred_dist=dataset.reserve_transpose_data(pred_dist)
        return pred_dist
        #return pred_seg, pred_dist

    def __make_3Dseg__(self, data, pred_Seg2D):
        if self.seg3D_connector:
            return self.seg3D_connector(data, pred_Seg2D)
        seg_connector = Simple_MaxCoverage_3DSegConnector()
        seg3d = seg_connector(data, pred_Seg2D)
        return seg3d


class Simple_MaxCoverage_3DSegConnector(object):
    # def __init__(self,data,seg2d):
    #   self.data = data
    #   self.seg2d =seg2d
    def __call__(self, data, seg2d):
        '''first, we need make sure that there are no same ids between slices''' 
        seg2d = self.reset_slice_id(seg2d)
        seg3d = self.update_sliceS_seg(seg2d, order='down')
        seg3d=self.update_sliceS_seg(seg3d,order ='up')
        return seg3d

    def reset_slice_id(self, seg2d):
        for i in range(len(seg2d)):
            seg2d[i] += (3000 * i)
        return seg2d

    def compute_iou(self, overlape_size, obj1_size, obj2_size, ):
        iou = float(overlape_size) / float(obj1_size + obj2_size)
        return iou

    def update_sliceS_seg(self, seg2d, order='down'):
        seg3d = seg2d.copy()
        max_IOU_cover_threshold = 0.15
        slice_idxs = range(len(seg2d)) if order == 'down' else range(len(seg2d))[::-1]
        for i in range(len(slice_idxs) - 1):
            ref_idx = slice_idxs[i]
            update_idx = slice_idxs[i + 1]
            seg_slice_1 = seg3d[ref_idx].copy()
            seg_slice_2 = seg3d[update_idx].copy()
            unique_ids_1, count_1 = np.unique(seg_slice_1, return_counts=True) 
            unique_ids_2, count_2 = np.unique(seg_slice_2, return_counts=True)
            s1_id_size = dict(zip(unique_ids_1, count_1))
            s2_id_size = dict(zip(unique_ids_2, count_2))
            idx = np.argsort(count_1)
            '''first connect to small objects, and then larger ones,
               This will ensure that larger object can overide to make final connections
               .which is crutial to IOU based measurement'''
            #sort_uids =unique_ids_1[idx]
            sort_uids = unique_ids_1[idx][::-1]
            ''' ----------------------------------------------------------------------- '''
            connected_ids = {sid:False for sid in unique_ids_2}
            max_cover_new_uids ={}
            for uid in sort_uids:
                bool_mask = (seg_slice_1 == uid)
                mask_ids = seg_slice_2[bool_mask]
                uids_2, count_uid = np.unique(mask_ids, return_counts=True)
                s2_cover_id_size = dict(zip(uids_2, count_uid))
                #idx = np.argmax(count_uid)
                #max_cover_id, max_cover_size = uids_2[idx], count_uid[idx]
                uid_size = s1_id_size[uid]
                ious = np.array([self.compute_iou(s2_cover_id_size[id], s2_id_size[id], uid_size) for id in uids_2])
                idxs = np.argsort(ious)
                # print(ious)
                uids_2 = uids_2[idxs][::-1]
                ious = ious[idxs][::-1]

                # refer_id_size = np.sum(bool_mask.astype(np.int))
                # max_cover_size = s2_id_size[max_cover_id]
                # refer_id_size = s1_id_size[uid]
                # if abs(0.5 - float(max_cover_size)/float((max_cover_size+refer_id_size))) < 0.2:
                #     seg3d[update_idx][seg3d[update_idx]==max_cover_id] =uid
                max_cover_id = uids_2[0]
                if ious[0] > max_IOU_cover_threshold:
                   if connected_ids[max_cover_id]:
                       seg3d[seg3d == max_cover_new_uids[max_cover_id]] = uid
                   else:
                      # seg3d[update_idx][seg3d[update_idx] == max_cover_id] = uid
                        seg3d[update_idx][seg_slice_2 == max_cover_id] = uid
                   connected_ids[max_cover_id] = True
                   max_cover_new_uids[max_cover_id] = uid
        return seg3d
                
def extend_tTensor_to_1250(vol):
    v_shape = vol.size()
    if v_shape[1] < 1250:
        vol_last = vol[:,-1:,:].repeat(1,1250-v_shape[1],1)
        vol = torch.cat([vol,vol_last],1)
    if v_shape[2] < 1250:
        vol_last = vol[:,:,-1:].repeat(1,1,1250-v_shape[2])
        vol = torch.cat([vol,vol_last],2)
    return vol
# def extendSeg_to_1250(vol):
#     v_shape = vol.shape
#     print(v_shape)
#     x_ext_slice_num = 1250 - v_shape[1]
#     y_ext_slice_num = 1250 - v_shape[2]
#     print(x_ext_slice_num)
#     x_s_list=[vol[::, -1:, ::].copy() for i in range(x_ext_slice_num)]
#     x_s_list.insert(0,vol)
#     seg_vol=np.concatenate(x_s_list, axis=1)

#     y_s_list=[vol[::, ::, -1:].copy() for i in range(y_ext_slice_num)]
#     y_s_list.insert(0,vol)
#     seg_vol=np.concatenate(y_s_list, axis=2)
#     return seg_vol


class em_seg_eval(object):
    def __init__(self, predict_config):
        self.exp_cfg = predict_config
        self.seg_predictor = em_seg_predict(self.exp_cfg)
    
    def predict(self):
        seg_conn_3d = {}
        seg_ws_3d = {}
        dist_front  = {}
        dist_combine = {}
        
        sub_datasets= self.exp_cfg.front_slice_dataset.subset
        print('subset data = {}'.format(sub_datasets))
        for d_set in sub_datasets:
            #d_set = 'Set_A'
            print('predicting {} ...'.format(d_set))
            self.exp_cfg.set_current_dataset(d_set)
            #seg_conn_p, seg_ws_p = self.seg_predictor.predict()
            #seg_ws_p = self.seg_predictor.predict()
            #seg_conn_3d[d_set]=extendSeg_to_1250(seg_conn_p)
            dist_front[d_set],dist_combine[d_set] = self.seg_predictor.predict()
        return dist_front, dist_combine

            #seg_ws_3d[d_set]=extendSeg_to_1250(seg_ws_p)
        #return seg_ws_3d
        #return seg_conn_3d, seg_ws_3d

    def eval(self):
        seg_conn_3d = {}
        seg_ws_3d  = {}
        dist_front  = {}
        dist_combine = {}
        seg_dict ={}
        arand_conn_eval = {}
        arand_ws_eval = {}
        # voi_eval = {}
        sub_datasets= self.exp_cfg.front_slice_dataset.subset
        print('subset data = {}'.format(sub_datasets))
        for d_set in sub_datasets:
            #d_set = 'Set_A'
            print('predicting {} ...'.format(d_set))
            self.exp_cfg.set_current_dataset(d_set)
            seg_lbs = self.exp_cfg.front_slice_dataset.get_label()
            #seg_conn_3d[d_set], seg_ws_3d[d_set] = self.seg_predictor.predict()
            #seg_ws_3d[d_set] = self.seg_predictor.predict()
            dist_front[d_set],dist_combine[d_set] = self.seg_predictor.predict()
            x_size = dist_front[d_set].shape[1]
            y_size = dist_front[d_set].shape[2]
            
            #x_size = seg_ws_3d[d_set].shape[1]
            #y_size = seg_ws_3d[d_set].shape[2]
            # x_size = seg_conn_3d[d_set].shape[1]
            # y_size = seg_conn_3d[d_set].shape[2]
            seg_lbs = seg_lbs[:, :x_size, :y_size]
            # arand_conn_eval[d_set] = adapted_rand(seg_conn_3d[d_set], seg_lbs)
            #arand_ws_eval[d_set] = adapted_rand(seg_ws_3d[d_set], seg_lbs)
            #print('arand conn for {} = {}'.format(d_set, arand_conn_eval[d_set]))
            #print('arand ws for {} = {}'.format(d_set, arand_ws_eval[d_set]))
            seg_dict[d_set] =seg_lbs

            #show_figure2(seg_conn_3d[d_set][10,:,:],seg_ws_3d[d_set][10,:,:])
            #voi_eval[d_set] = voi(seg_3d[d_set],seg_lbs)
            #show_figure(seg_ws_3d[d_set], slice_idx = 0)
            void_eval = 0
        return dist_front, dist_combine, seg_dict
        #return arand_conn_eval, void_eval

def show_figure(seg3D, mode = 'segment', slice_idx = 0):
        my_dpi = 96
        fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
        if mode =='segment':
            label2d_seg = label2rgb(seg3D)    
            a = fig.add_subplot(1, 2, 1)
            plt.imshow(label2d_seg[0], interpolation='nearest')
            a.set_title('upper_seg')    
            a = fig.add_subplot(1, 2, 2)
            plt.imshow(label2d_seg[1], interpolation='nearest')
            a.set_title('lower_seg')
        elif mode =='original':
            #a = fig.add_subplot(1, 2, 1)
            plt.imshow(np.squeeze(seg3D[slice_idx]))
            #a.set_title('upper_seg')    
            #a = fig.add_subplot(1, 2, 2)
            #plt.imshow(label2d_seg[1], interpolation='nearest')
            #a.set_title('lower_seg')

        plt.show()

def show_figure2(seg_p,seg_g):
    my_dpi = 96
    fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
    label2d_seg_p = label2rgb(seg_p)    
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(label2d_seg_p, interpolation='nearest')
    a.set_title('p_seg')  
    label2d_seg_g = label2rgb(seg_g)  
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(label2d_seg_g, interpolation='nearest')
    a.set_title('g_seg')
    plt.show()
