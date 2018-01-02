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
        
        # create network and load the weights
        net_model = NETWORKS[self.net_conf['model']]
        data_out_labels = self.dataset.output_labels()
        self.network = net_model(target_label=data_out_labels, in_ch=self.net_conf['z_slices'])
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
        pred_seg_3d = self.__make_3Dseg__(self.dataset.get_data(),pred_seg_2d)
        return pred_seg_3d



    def __predict2D__(self):
        if self.use_gpu:
            print ('model_set_cuda')
            self.model = self.model.cuda().float()
        self.model.eval()
        pred_seg=torch.zeros_like(torch.from_numpy(self.dataset.get_data()))
        pred_seg.float()
        p_shape = pred_seg.shape
        print('pred_seg shape {}'.format(pred_seg.shape))
        pred_seg.resize_(p_shape[0],1248,1248)
        raw_Data =self.dataset.get_data()
        #g_seg    =self.dataset.get_label()
        print('dataset len = {}'.format(len(self.dataset)))
        for i in range(len(self.dataset)):
            out = self.dataset.__getitem__(i)
            # d = out['data'][0,1].clone()
            # plt.imshow(d.cpu().numpy())
            # plt.show()

            #out_data = out['data'].resize_(1,3,1248,1248)
            '''resize_ will result in deformed image, interesting!
               we may be able to use this function as  a mean of augmentation'''

            # plt.imshow(out_data[0,1].cpu().numpy())
            # plt.show()
            out_data    = out['data'][:,:,:1248,:1248]
            #print(out['label'])
            g_seg_data  =out['label'][:,:,:1248,:1248]
            #print('out_data shape = {}'.format(out_data) )

            data = Variable(out_data).float()
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
            fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
            
            a = fig.add_subplot(3, 1, 1)
            d =np.squeeze(out_data.cpu().numpy())
            print('d shape = {}'.format(d.shape))
            imgplot = plt.imshow(d[1])
            a.set_title('im')
            
            a = fig.add_subplot(3, 1, 2)
            plt.imshow(label2rgb(watershed_d.astype(np.int)), interpolation='nearest')
            a.set_title('p_seg')

            a = fig.add_subplot(3, 1, 3)
            plt.imshow(label2rgb(np.squeeze(g_seg_data.cpu().numpy())), interpolation='nearest')
            a.set_title('g_seg')
            
            plt.savefig('watershed_d_{}.png'.format(i))
            plt.close()
            centermap = np.squeeze(preds['centermap'].data.cpu().numpy())

            
            #print('centermap shape = {}'.format(centermap.shape))

            # plt.imshow(centermap[0])
            # plt.savefig('centerX_{}.png'.format(i))
            # plt.close()

            # plt.imshow(centermap[1])
            # plt.savefig('centerY_{}.png'.format(i))
            # plt.close()

            # #plt.imshow(out['data'][0,1].cpu().numpy())
            # plt.imshow(raw_Data[i])
            # plt.savefig('raw_{}.png'.format(i))
            # plt.close()





            pred_seg[i]= torch.from_numpy(watershed_d)
        return pred_seg

    def __make_3Dseg__(self, data, pred_Seg2D):
        if self.seg3D_connector:
            return self.seg3D_connector(data, pred_Seg2D)
        pred_Seg2D = pred_Seg2D.cpu().numpy()
        seg_connector =Simple_MaxCoverage_3DSegConnector()
        seg3d = seg_connector(data,pred_Seg2D)
        return seg3d


class Simple_MaxCoverage_3DSegConnector(object):
    # def __init__(self,data,seg2d):
    #   self.data = data
    #   self.seg2d =seg2d
    def __call__(self,data,seg2d):
        seg_3d = seg2d.copy()
        for i in range(len(seg2d)-1):
            seg_slice_1 = seg2d[i]
            seg_slice_2 = seg2d[i+1]

            unique_ids_1,count_1 = np.unique(seg_slice_1,return_counts = True)
            unique_ids_2,count_2 = np.unique(seg_slice_2,return_counts = True)
            #print(unique_ids_1)
            sort_uids = np.sort(unique_ids_1)[::-1]
            #sort_uids = sort_uids[:,:-1]
            connected_ids ={sid:False for sid in unique_ids_2}

            for uid in sort_uids:
                bool_mask = (seg_slice_1==uid)
                mask_ids = seg_slice_2[bool_mask]
                uids_2, count = np.unique(mask_ids, return_counts=True)
                idx = np.argmax(count)
                max_cover_id = uids_2[idx]
                if connected_ids[max_cover_id]:
                    seg_3d[seg_3d==max_cover_id]=uid
                else:
                    s=seg_3d[i]
                    s[s==max_cover_id] =uid
                    seg_3d[i] = s
                    connected_ids[uid]=True

        return seg_3d
                


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
            print('predicting {} ...'.format(d_set))
            self.exp_cfg.dataset.set_current_subDataset(d_set)
            seg_lbs = self.exp_cfg.dataset.get_label()
            seg_3d[d_set]=self.seg_predictor.predict()
            x_size = seg_3d[d_set].shape[1]
            y_size = seg_3d[d_set].shape[2]
            seg_lbs= seg_lbs[:,:x_size,:y_size]
            arand_eval[d_set]=adapted_rand(seg_3d[d_set],seg_lbs)
            #voi_eval[d_set] = voi(seg_3d[d_set],seg_lbs)
            void_eval = 0
        return arand_eval, voi_eval
