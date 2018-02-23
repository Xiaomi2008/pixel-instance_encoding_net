from utils.EMDataset import CRIME_Dataset3D
import pytoml as toml
import torch
from torch_networks.res_3D2Dhybrid_unet \
     import hybrid_2d3d_unet_mutlihead, hybrid_2d3d_unet
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb 
from utils.utils import watershed_seg,evalute_pred_dist_with_threshold
#from utils.evaluation import evalute_pred_dist_with_threshold
import pdb

class predict_config():
  def __init__(self, config_file):
      def build_network_and_dataset(net_conf, label_conf, patch_size, overlap_size):
            data_set = CRIME_Dataset3D(sub_dataset=self.dataset_conf['sub_dataset'],
                                       subtract_mean=True,
                                       label_config =label_conf['labels'],
                                       phase ='valid',
                                       predict_patch_size = patch_size,
                                       predict_overlap = overlap_size,
                                       data_config=data_config)


            #pdb.set_trace()

            data_out_labels = data_set.output_labels()
            # input_lbCHs_cat_for_net2 = label_conf['label_catin_net2']

            # initial between network and load weight
            net_model = NETWORKS[net_conf['model']]
            network = net_model(target_label=data_out_labels, in_ch=1, BatchNorm_final=False)
            pre_trained_file = net_conf['trained_file']
            print('load weights from {}'.format(pre_trained_file))
            network.load_state_dict(torch.load(pre_trained_file))
            return network, data_set

      self.parse_toml(config_file)
      NETWORKS = \
              {'res_3D2D_hybrid_net': hybrid_2d3d_unet,
               'res_3D2D_hybrid_net_multi_output': hybrid_2d3d_unet_mutlihead}

      dset =self.dataset_conf['dataset']
      if  dset== 'predict':
         data_config = 'conf/cremi_datasets_test.toml'
         split = 'predict'
      elif dset == 'valid':
         data_config = 'conf/cremi_datasets.toml'
         split = 'valid'
      else:
        raise ValueError('unknow dataset for {}'.format(dset))

      
      #pdb.set_trace()
      # need to reserve the oder of path_size, in the toml the order is x,y,z. but 
      # in the numpy array or hdf5 its order is z,y,x
      self.predict_conf['preidct_patch_size'].reverse()
      self.predict_conf['predict_overlap'].reverse()

      self.network, self.dataset = build_network_and_dataset(self.net_conf,self.label_conf,
                                                            self.predict_conf['preidct_patch_size'],
                                                            self.predict_conf['predict_overlap'])

        

  def parse_toml(self, file):
        with open(file, 'rb') as fi:
            conf = toml.load(fi)
            net_conf = conf['network']
            net_conf['model'] = net_conf.get('model', 'res_3D2D_hybrid_net_multi_output')
            label_conf = conf['target_labels']
            label_conf['labels'] = label_conf.get('labels',
                                                  ['affinityX', 'affinityY', 'affinityZ'])
            predict_conf = conf['predict_setting']
            predict_conf['predict_patch_size'] =predict_conf.get('predict_patch_size',[320,320,25])
            predict_conf['predict_overlap'] =predict_conf.get('predict_overlap',[60,60,8])

            self.net_conf = net_conf
            self.label_conf = label_conf
            self.predict_conf =predict_conf
            self.dataset_conf = conf['dataset']
            self.conf = conf



class em_seg_predict():
    def __init__(self, predict_config):
        self.exp_cfg = predict_config
        self.use_gpu = self.exp_cfg.net_conf['use_gpu'] and torch.cuda.is_available()
        self.model = self.exp_cfg.network
        self.dataset = self.exp_cfg.dataset

    def predict(self,dset_name = 'Set_A'):
        self.dataset.set_current_subDataset(dset_name)
        pred_dict = self.__predict__()
       
        return pred_dict
        #pred_seg_3dWS  = watershed_seg(pred_dist_2d,threshold=1.5)

        #return pred_seg_3dconn, pred_seg_3dWS

    def eval(self,dset_name ='Set_A'):
        pred_dict = self.predict(dset_name)
        seg_lbs = self.exp_cfg.dataset.get_label()
        #input_d =(pred_dict['distance2D'][1]+pred_dict['distance3D'][0])/2.0

        #pdb.set_trace()
        #evalute_pred_dist_with_threshold(seg_lbs[100:,:,:],input_d[100:,:,:])

        return pred_dict, seg_lbs


    def __predict__(self):
        def tensor_dict_to_numpy(preds):
          out={}
          for k,v in preds.iteritems():
            #if k =='affinityX' or k=='affinityY' or k=='affinityZ':
            if k ==['affinityX','affinityY','affinityZ']:
                v = torch.sigmoid(v)
            out[k] = v.data.cpu().numpy()
          return out

        if self.use_gpu:
          print ('model_set_cuda')
          self.model = self.model.cuda().float()
        self.model.eval()

        # slicing large data valume into small valumes
        # returns: (sliced_data_dict) contain a dict of array list,
        #           (slice_obj_list) slice oject list which record where the slice occured
        #           (overlap_count_array) count the voxel-wise slice overlap
        sliced_data_dict, slice_obj_list, overlap_count_array = \
          self.dataset.conv_slice_3DPatch()

        
        # prdict each slice of data. Note that affinity has mutiple channels, each represent 
        # a different boundary distance
        predict_dict_slice_data_list = {l:[] for l in self.exp_cfg.label_conf['labels']}
        for idx, data in enumerate(sliced_data_dict['image']):
          d1 = np.expand_dims(data,axis=0).copy()
          d1 = np.expand_dims(d1,axis=0)
          d1 = torch.from_numpy(d1).cuda().float()
          print('predict slice {}, input size {}'.format(idx, d1.size()))
          input_data = Variable(d1, volatile=True)
          #input_data = input_data.cuda() if self.use_gpu else input_data
          #print('predicting slice one')
          preds = self.model(input_data)
          #print('pred one')
          preds = tensor_dict_to_numpy(preds)
          for k,v in preds.iteritems():
            predict_dict_slice_data_list[k].append(v)

        assembled_pred_data_dict ={}
        for k, data_list in predict_dict_slice_data_list.iteritems():
          assembled_pred_data_dict[k]= \
             self.dataset.assemble_conv_slice(data_list, slice_obj_list, overlap_count_array)
        return assembled_pred_data_dict