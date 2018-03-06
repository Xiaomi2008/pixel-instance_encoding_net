# from experiment import experiment_config
from utils.EMDataset import slice_dataset
from utils.utils import watershed_seg, rondomwalker_seg, watershed_on_distance_and_skeleton
from utils.slice_connector import Simple_MaxCoverage_3DSegConnector
from utils.evaluation import adapted_rand
from misc.orig_cremi_evaluation import voi
#from utils.slice_connector import slice_3D_connector
import pytoml as toml
import torch
from torch_networks.networks import Unet, DUnet, MdecoderUnet, Mdecoder2Unet, \
    MdecoderUnet_withDilatConv, Mdecoder2Unet_withDilatConv,MdecoderUnet_withFullDilatConv
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt

import pdb
from misc.orig_cremi_evaluation import voi
from skimage.measure import label
from skimage.color import label2rgb
from skimage.filters import gaussian
#import pdb
class predict_config():
    def __init__(self, config_file):
        self.parse_toml(config_file)
        NETWORKS = \
            {'Unet': Unet, 'DUnet': DUnet, 'MDUnet': MdecoderUnet,
             'MDUnetDilat': MdecoderUnet_withDilatConv, 'M2DUnet': Mdecoder2Unet,
             'M2DUnet_withDilatConv': Mdecoder2Unet_withDilatConv,
             'MDUnet_FullDilat':MdecoderUnet_withFullDilatConv}

        if self.dataset_conf['dataset'] == 'valid':
            data_config = 'conf/cremi_datasets.toml'
            split = 'valid'
        elif self.dataset_conf['dataset'] == 'predict':
            data_config = 'conf/cremi_datasets_test.toml'
            split = 'predict'
        else:
            data_config = 'conf/cremi_datasets.toml'
            split = 'valid'


        '''' create dataset which is able to iteratively obtain slices of image 
        (3,5 slice with stride 1) from either z-direction ir xy-direction
        '''
        self.dataset = slice_dataset(sub_dataset=self.dataset_conf['sub_dataset'],
                                     subtract_mean=True,
                                     split=split,
                                     slices=self.net_conf['z_slices'],
                                     data_config=data_config)
                                #data_config='conf/cremi_datasets_with_tflabels.toml')
        data_out_labels = self.dataset.output_labels()
        input_lbCHs_cat_for_net2 = self.label_conf['label_catin_net2']
        
        net1_out_put_label = self.label_conf['labels']

        
        net1_target_label_ch_dict={}
        for lb,ch in data_out_labels.iteritems():
            if lb in net1_out_put_label:
                net1_target_label_ch_dict[lb] =ch


        if 'final_labels' in self.label_conf:
            net2_out_put_label=self.label_conf['final_labels']
            net2_target_label_ch_dict= {}
            for lb,ch in data_out_labels.iteritems():
                if lb in net2_out_put_label:
                    net2_target_label_ch_dict[lb] =ch
        elif 'final_label' in self.label_conf:
            net2_target_label_ch_dict= {}
            net2_target_label_ch_dict['final']=data_out_labels[self.label_conf['final_label']]




        #target_label_ch_dict ={ lb:ch if lb in network_out_put_label for lb,ch in data_out_labels}

        print(data_out_labels)
        label_ch_pair_info ={'gradient':2,'sizemap':1,'affinity':1,'centermap':2,'distance':1,'skeleton':1}
        # if 'sub_net' in self.conf:
        #     net_1_ch_pair = {}
        #     for lb in self.label_conf['labels']:
        #         net_1_ch_pair[lb] = label_ch_pair_info[lb]
        #     subnet_model = NETWORKS[self.conf['sub_net']['model']]
        #     self.sub_network = subnet_model(target_label=net_1_ch_pair, in_ch=self.net_conf['z_slices'])


        # self.network = net_model(self.sub_network, 
        #                              freeze_net1=freeze_net1,
        #                              target_label=net_1_ch_pair,
        #                              net2_target_label= label_ch_pair,
        #                              label_catin_net2=input_lbCHs_cat_for_net2,
        #                              in_ch=in_ch,
        #                              out_ch=out_ch,
        #                              first_out_ch=16)

        # create network and load the weights
        net_model = NETWORKS[self.net_conf['model']]
        if self.net_conf['model'] == 'M2DUnet_withDilatConv':
            self.network = net_model(freeze_net1=True,
                                     target_label=net1_target_label_ch_dict,
                                     net2_target_label= net2_target_label_ch_dict,
                                     label_catin_net2=input_lbCHs_cat_for_net2,
                                     in_ch=self.net_conf['z_slices'])
            print(net_model)
        else:
            print(self.net_conf['model'])
            self.network = net_model(target_label=net1_target_label_ch_dict, in_ch=self.net_conf['z_slices'],BatchNorm_final=False)
        #self.network = net_model(target_label=data_out_labels, in_ch=self.net_conf['z_slices'])


        self.use_gpu =True
        if self.use_gpu and torch.cuda.is_available():
            print ('model_set_cuda')
            self.network = self.network.cuda()
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
    def __init__(self, predict_config, seg3D_connector=None):
        self.exp_cfg = predict_config
        self.use_gpu = self.exp_cfg.net_conf['use_gpu'] and torch.cuda.is_available()
        self.model = self.exp_cfg.network
        self.dataset = self.exp_cfg.dataset
        self.seg3D_connector = seg3D_connector


    def predict(self,required_outputs=['all']):
        preds = self.__predict2D__(required_outputs=required_outputs)
        return preds

    def __predict2D__(self, required_outputs=['all'],direction ='z_axis'):
        def get_network_output_list():
            im_label = self.dataset.__getitem__(0)
            input_im = im_label['data'][:,:,:cut_size,:cut_size]
            data = Variable(input_im,volatile=True).float()
            data = data.cuda().float() if self.use_gpu else data
            preds = self.model(data)
            #predict_names =[]
            #map(filter(lambda (k,v):v.shape[1]==1, preds)
            # we filter out output that has more then 1 channel 
            # as we currently unable to handle it
            predict_names=[k for (k,v) in preds.iteritems() if v.shape[1] ==1]
            return predict_names
        
        cut_size = 1248
        out_names=get_network_output_list() \
                                           if 'all' in required_outputs \
                                           else required_outputs

        d_shape=self.dataset.get_data().shape
        out_data_dict ={out_name:torch.zeros(d_shape) for out_name in out_names}

        for k,v in out_data_dict.iteritems():
            out_data_dict[k] = v[:,:cut_size,:cut_size] \
                                if direction =='z_axis' \
                                else v[:,::,:cut_size]
        
        if self.use_gpu:
            print ('model_set_cuda')
            self.model = self.model.cuda().float()
        self.model.eval()

        for i in range(len(self.dataset)):
            out = self.dataset.__getitem__(i)
            out_data = out['data'][:,:,:cut_size,:cut_size]
            data = Variable(out_data,volatile=True).float()
            data = data.cuda().float() if self.use_gpu else data.float()
            preds = self.model(data)
            for k,v in out_data_dict.iteritems():
                out_data_dict[k][i,:,:] = preds[k].data.cpu()


        return out_data_dict
      



    # def predict(self, net_out_only =False):
    #     pred_seg_2d, pred_dist_2d = self.__predict2D__()
    #     print('connecting slices ids ...')
    #     image = self.dataset.get_data()
    #     pred_seg_3dconn= self.__make_3Dseg__(image, pred_seg_2d.cpu().numpy())
    #     pred_seg_3dWS  = watershed_seg(pred_dist_2d.cpu().numpy(),threshold=0.11)
    #     #pdb.set_trace()

    #     return pred_seg_3dconn, pred_seg_3dWS, image

    # def __predict2D__(self, direction ='low_res',net_out_only=False):


    #     out_only_data_names =['distance', 'final']
    #     # if self.use_gpu:
    #     #     print ('model_set_cuda')
    #     #     self.model = self.model.cuda().float()

    #     self.model.eval()
    #     pred_seg = torch.zeros_like(torch.from_numpy(self.dataset.get_data().astype(int)))
    #     pred_seg = pred_seg.long()
    #     p_shape  = pred_seg.shape
    #     pred_dist = torch.zeros_like(torch.from_numpy(self.dataset.get_data().astype(int)))
    #     cut_size = 1248
    #     print('pred_seg shape {}'.format(pred_seg.shape))
    #     if direction == 'low_res':
    #          #pred_seg = pred_seg[:p_shape[0],::,:cut_size]
    #          #pred_dist = pred_dist[:p_shape[0],::,:cut_size]
    #          pred_seg = pred_seg[:p_shape[0],:cut_size,:cut_size]
    #          pred_dist = pred_dist[:p_shape[0],:cut_size,:cut_size]
    #          #pred_seg = np.transpose(pred_seg,[])
    #     else:
    #          pred_seg = pred_seg[:p_shape[0],:cut_size,:cut_size]
    #          pred_dist = pred_dist[:p_shape[0],:cut_size,:cut_size]

    #     for i in range(len(self.dataset)):
    #         out = self.dataset.__getitem__(i)
    #         out_data = out['data'][:,:,:cut_size,:cut_size]
    #         g_seg_data = None
    #         if 'label' in out:
    #             #g_seg_data = out['label'][:,:,::,:cut_siz]
    #             g_seg_data = out['label'][:,:,:cut_size,:cut_size]
    #         data = Variable(out_data,volatile=True).float()
    #         if self.use_gpu:
    #             data = data.cuda().float()

    #         preds = self.model(data)
    #         if not net_out_only:
    #             if 'skeleton' in self.exp_cfg.label_conf['labels']\
    #                 or 'final_labels' in self.exp_cfg.label_conf \
    #                 and 'skeleton' in self.exp_cfg.label_conf['final_labels']:
    #                 dis=np.squeeze(preds['distance'].data.cpu().numpy())
    #                 sk=np.squeeze(torch.sigmoid(preds['skeleton']).data.cpu().numpy())
    #                 #show_2figure(dis,sk)
    #                 watershed_d = watershed_on_distance_and_skeleton(preds['distance'], torch.sigmoid(preds['skeleton']))
    #                 distance_d  = preds['distance'].data.cpu()
    #                 #pdb.set_trace()
    #             else:
    #                 watershed_d = np.squeeze(watershed_seg(preds['distance'],threshold=0.06))
    #                 #distance_d  = preds['distance'].data.cpu()
    #                 distance_d  = preds['distance'].data.cpu()
    #                 #show_2figure(np.squeeze(watershed_d),np.squeeze(distance_d.numpy()),first_is_seg =True)
    #                 #watershed_d = np.squeeze(rondomwalker_seg(preds['final'],threshold=7))

                    


    #             #pdb.set_trace()
    #             #plt.imshow(np.squeeze(preds['distance'].data.cpu().numpy()))
    #             #plt.show()
    #             #show_2figure(watershed_d,np.squeeze(preds['final'].data.cpu().numpy()))


    #             #watershed_d = np.squeeze(watershed_seg((preds['final'] + preds['distance'])/2.0))

    #             #distance_d  = (preds['final'].data +  preds['distance'].data)/2.0

    #             #my_dpi = 96
    #             if g_seg_data is not None:
    #                 # output is a 3 slice data in channels, but we only need the cetner [1]
    #                 g_seg_in = np.squeeze(g_seg_data.cpu().numpy())[1] 
    #                 arand_eval = adapted_rand(watershed_d.astype(np.int), g_seg_in)
    #                 (split,merge) =voi(watershed_d.astype(np.int), g_seg_in)

    #                 print('arand :{} (split, merge) = ({},{}) '.format(arand_eval,split,merge))

    #                 relabel_gseg = label(g_seg_in)

    #                 arand_eval = adapted_rand(watershed_d.astype(np.int), relabel_gseg)
    #                 (split,merge) =voi(watershed_d.astype(np.int), relabel_gseg)

    #                 print('relabeld :  arand :{} (split, merge) = ({},{}) '.format(arand_eval,split,merge))

    #             pred_seg[i,:,:]= torch.from_numpy(watershed_d)
    #             pred_dist[i,:,:]= distance_d
            
    #         else:
    #             if 'distance' 

    #     return pred_seg, pred_dist

    def __make_3Dseg__(self, data, pred_Seg2D):
        if self.seg3D_connector:
            return self.seg3D_connector(data, pred_Seg2D)
        seg_connector = Simple_MaxCoverage_3DSegConnector()
        seg3d = seg_connector(data, pred_Seg2D)
        return seg3d

    #def __make_3D_watershed__(self,data,pred_Bnd2D):




                

def extendSeg_to_1250(seg_vol):
    v_shape = seg_vol.shape
    print(v_shape)
    x_ext_slice_num = 1250 - v_shape[-2]
    y_ext_slice_num = 1250 - v_shape[-1]
    print(x_ext_slice_num)
    x_s_list=[seg_vol[::, -1:, ::].copy() for i in range(x_ext_slice_num)]
    x_s_list.insert(0,seg_vol)
    seg_vol=np.concatenate(x_s_list, axis=1)

    y_s_list=[seg_vol[::, ::, -1:].copy() for i in range(y_ext_slice_num)]
    y_s_list.insert(0,seg_vol)
    seg_vol=np.concatenate(y_s_list, axis=2)
    return seg_vol


class em_seg_eval(object):
    def __init__(self, predict_config):
        self.exp_cfg = predict_config
        #self.seg_predictor = em_seg_predict(self.exp_cfg,seg3D_connector =slice_3D_connector())
        self.seg_predictor = em_seg_predict(self.exp_cfg,seg3D_connector =None)
    
    def predict(self,set_name=None,required_outputs=['all']):
        #seg_conn_3d = {}
        #seg_ws_3d = {}

        pred_set_names = set_name if set_name else self.exp_cfg.dataset.subset
        pred_dict ={}

        print('subset data = {}'.format(pred_set_names))
        #print('subset data = {}'.format(self.exp_cfg.dataset.subset))
        for d_set in pred_set_names:
            #d_set = 'Set_A'
            print('predicting {} ...'.format(d_set))
            self.exp_cfg.dataset.set_current_subDataset(d_set)
            preds = self.seg_predictor.predict(required_outputs=required_outputs)
            # simply duplicate last few slices to make data 1250 res
            # and also convert for torch tensor to numpy()
            for k,v in preds.iteritems():
                preds[k] = extendSeg_to_1250(v.numpy())

            pred_dict[d_set] =preds
        return pred_dict
            
            #seg_conn_p, seg_ws_p, im = self.seg_predictor.predict()
            #seg_conn_3d[d_set]=extendSeg_to_1250(seg_conn_p)
            #seg_ws_3d[d_set]=extendSeg_to_1250(seg_ws_p)
            
            #print('subimssion conn_seg {} shape {}'.format(d_set, seg_conn_3d[d_set].shape))
            #print('subimssion sw_seg {} shape {}'.format(d_set, seg_ws_3d[d_set].shape))
            #self.show_figure( seg_3d[d_set])
        #return seg_conn_3d, seg_ws_3d, im

    def eval(self, set_name=None):
        seg_conn_3d = {}
        seg_ws_3d  = {}
        arand_conn_eval = {}
        arand_ws_eval = {}
        # voi_eval = {}
        eval_sets = [set_name] if set_name else self.exp_cfg.dataset.subset

        print('subset data = {}'.format(eval_sets))
        for d_set in eval_sets:
            #d_set = 'Set_A'
            print('predicting {} ...'.format(d_set))
            self.exp_cfg.dataset.set_current_subDataset(d_set)
            seg_lbs = self.exp_cfg.dataset.get_label()
            preds  = self.predict(d_set)
            # seg_conn_3d[d_set], seg_ws_3d[d_set] = self.seg_predictor.predict()
            # x_size = seg_conn_3d[d_set].shape[1]
            # y_size = seg_conn_3d[d_set].shape[2]
            # seg_lbs = seg_lbs[:, :x_size, :y_size]
            arand_conn_eval[d_set] = adapted_rand(seg_conn_3d[d_set], seg_lbs)
            # arand_ws_eval[d_set] = adapted_rand(seg_ws_3d[d_set], seg_lbs)
            # voi_conn_eval[d_set] = voi(seg_conn_3d[d_set], seg_lbs)
            # voi_ws_eval[d_set] = voi(seg_ws_3d[d_set], seg_lbs)
            print('arand,voi conn for {} = {},{}'.format(d_set, arand_conn_eval[d_set],voi_conn_eval[d_set]))
            print('arand. voi ws for {} = {},{}'.format(d_set, arand_ws_eval[d_set],voi_ws_eval[d_set]))

        return arand_conn_eval, voi_conn_eval

    def show_figure(self, seg3D):
        my_dpi = 96
        fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
        label2d_seg = label2rgb(seg3D)    
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(label2d_seg[0], interpolation='nearest')
        a.set_title('upper_seg')    
        a = fig.add_subplot(1, 2, 2)
        plt.imshow(label2d_seg[1], interpolation='nearest')
        a.set_title('lower_seg')
        plt.show()


def show_2figure(seg_p,im,first_is_seg =False):
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    fig, axes = plt.subplots(nrows=1, ncols=4, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    #label2d_seg_p = label2rgb(seg_p)
    if first_is_seg:
        axes[0].imshow(np.random.permutation(seg_p.max() + 1)
                                    [seg_p],
                                    cmap='spectral') 
    else:
        seg_p =gaussian(seg_p / float(np.max(seg_p)), sigma=0.6, mode='reflect')    
        axes[0].imshow(seg_p, interpolation='nearest')
    axes[0].set_title('distance')
    axes[0].axis('off')
    axes[0].margins(0, 0)


    im=gaussian(im / float(np.max(im)), sigma=0.6, mode='reflect')  
    
    axes[1].imshow(im, interpolation='nearest')
    axes[1].set_title('skeleton')
    axes[1].axis('off')
    axes[1].margins(0, 0)


    # image_max = ndi.maximum_filter(im, size=5, mode='constant')
    # coordinates = peak_local_max(image_max, min_distance=30)

    # axes[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')

    #seg_p =gaussian(seg_p / float(np.max(seg_p)), sigma=0.6, mode='reflect') 
    #axes[2].imshow((im>0.06).astype(int), interpolation='nearest')
    merge = seg_p + im
    axes[2].imshow(merge)
    axes[2].set_title('distance + skeleton')
    axes[2].axis('off')
    axes[2].margins(0, 0) 
    #plt.show()


   
    #pdb.set_trace()
    #axes[2].imshow((im>0.06).astype(int), interpolation='nearest')
    axes[3].imshow(merge>0.06)
    axes[3].set_title('threshold mask')
    axes[3].axis('off')
    axes[3].margins(0, 0) 
    plt.show()


def visualize_mask_image(seg_mask,im,lb ,idx,connected):
  fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
  axes[0, 0].imshow(im[0], cmap='gray')
  axes[0, 0].axis('off')
  axes[0, 0].margins(0, 0)
  title_str = 'prob : {}, gt : {}'.format(lb,connected)
  axes[0,0].set_title(title_str)

  axes[0, 1].imshow(im[1], cmap='gray')
  axes[0, 1].axis('off')
  axes[0, 1].margins(0, 0)

  axes[1, 0].imshow(seg_mask[0])
  axes[1, 0].axis('off')
  axes[1, 0].margins(0, 0)

  axes[1, 1].imshow(seg_mask[1])
  axes[1, 1].axis('off')
  axes[1, 1].margins(0, 0)
  plt.savefig('mask_test{}.png'.format(idx))
  plt.show()



def show_figure2(seg_p,seg_g):
    my_dpi = 96
    #fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)

    fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    label2d_seg_p = label2rgb(seg_p)    
    axes[0].imshow(label2d_seg_p, interpolation='nearest')
    axes[0].set_title('p_seg')
    axes[0].axis('off')
    axes[0].margins(0, 0)  
    
    label2d_seg_g = label2rgb(seg_g)  
    axes[1].imshow(label2d_seg_g, interpolation='nearest')
    axes[1].set_title('g_seg')
    axes[1].axis('off')
    axes[1].margins(0, 0) 
    plt.show()
