import argparse
import os
import numpy as np
from experiment import experiment_config, experiment
from MaskNet_experiment import masknet_experiment, masknet_experiment_config
from sliceConnNet_experiment import slice_connect_experiment_config,\
                             slice_connect_experiment
#from eval_predict import predict_config, em_seg_eval
#from eval_predict_3d_slice import predict_config, em_seg_eval
from eval_predict3D import predict_config, em_seg_predict
from cremi.io import CremiFile
from cremi import Annotations, Volume
from utils.evaluation import adapted_rand, voi
import h5py

def make_seg_submission(seg_valume_dict):
    submission_folder = 'submission'
    if not os.path.exists(submission_folder):
        os.makedirs(submission_folder)
    for name, seg_v in seg_valume_dict.iteritems():
        seg_v = seg_v.astype(np.uint64)
        neuron_ids = Volume(seg_v, resolution=(40.0, 4.0, 4.0), comment="First submission in 2018")
        file = CremiFile(submission_folder+'/'+name + '.hdf', "w")
        file.write_neuron_ids(neuron_ids)

def main():
    parser = argparse.ArgumentParser(description='main arguments')
    parser.add_argument('-a', '--action', type=str, default='Train2D',
                        help='Option: train2D, train_mask, evaluation, predict')
    parser.add_argument('-e', '--exp_cfg', type=str, default='conf/experiment_1.toml',
                        help='path of the experiment_config file')
    args = parser.parse_args()

    if args.action == 'train2D':
        exp_cfg = experiment_config(args.exp_cfg)
        print("=======================")
        exp_obj = experiment(exp_cfg)
        exp_obj.train()

    elif args.action == 'evaluation':
        # print('evaluation')
        # exp_cfg = predict_config(args.exp_cfg)
        # exp_obj = em_seg_eval(exp_cfg)
        # d1,d2,t1 =exp_obj.eval()
        # save_eval_dist(d1,d2,t1)

        # return d1,d2,t1
        print('predicting ...')
        exp_cfg = predict_config(args.exp_cfg)
        #exp_obj = em_seg_eval(exp_cfg)
        exp_obj = em_seg_predict(exp_cfg)
        d1,seg_t = exp_obj.eval('Set_C')
        return d1, seg_t



    elif args.action == 'predict':
        print('predicting ...')
        exp_cfg = predict_config(args.exp_cfg)
        #exp_obj = em_seg_eval(exp_cfg)
        exp_obj = em_seg_predict(exp_cfg)
        #d1,seg_t = exp_obj.eval('Set_B')
        d1  =exp_obj.predict('Set_B')
        return d1, None
        #d1,d2 =exp_obj.predict()
        #save_prict_dist(d1,d2)




        #seg3D_dict = exp_obj.predict()
        #make_seg_submission(seg3D_dict)
        # print(arand)
        # print(voi)
    elif args.action == 'train_mask':
        exp_cfg = masknet_experiment_config(args.exp_cfg)
        exp_obj = masknet_experiment(exp_cfg)
        exp_obj.train()

    elif args.action == 'train_obj_connection_classifier':
        exp_cfg = slice_connect_experiment_config(args.exp_cfg)
        exp_obj = slice_connect_experiment(exp_cfg)
        exp_obj.train()



def save_prict_dist(d1,d2):
    h5f=h5py.File('tempdata/predict_seg_final_plus_distance.h5', 'w')
    for set_name in d1.keys():
        h5f.create_dataset(set_name + '_d1', data = d1[set_name])
        h5f.create_dataset(set_name + '_d2', data = d2[set_name])
    h5f.close()

def save_eval_dist(d1,d2,t1):
    h5f=h5py.File('tempdata/seg_final_plus_distance.h5', 'w')
    for set_name in d1.keys():
        h5f.create_dataset(set_name + '_d1', data = d1[set_name])
        h5f.create_dataset(set_name + '_d2', data = d2[set_name])
        h5f.create_dataset(set_name + '_t1', data = t1[set_name])
    h5f.close()

def save_view(pred_dict,seg_t=None):
    from skimage.color import label2rgb
    from matplotlib import pyplot as plt
    import torch
    # td2=torch.sigmoid(torch.from_numpy(pred_dict['distance2D'][0]))
    # nd2=td2.numpy()
    nd2 = pred_dict['distance2D'][0]

    plt.imshow(nd2[:,100,:])
    plt.savefig('distance2D_setA_hsection_100_valid.png')

    plt.imshow(nd2[105,:,:])
    plt.savefig('distance2D_setA_xy_105_valid.png')

    if seg_t is not None:
        plt.imshow(label2rgb(seg_t[:,100,:]))
        plt.savefig('segT_setA_hsection_100_valid.png')
        plt.imshow(seg_t[105,:,:])
        plt.savefig('segT_setA_xy_105_valid.png')


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '8'
    main()
    # pred_dict,seg_t=main()
    # from utils.utils import watershed_seg,evalute_pred_dist_with_threshold
    # input_d =pred_dict['distance3D'][0]
    # thresholds = np.linspace(1.45,2,10)
    # #evalute_pred_dist_with_threshold(np.array(seg_t),input_d,thresholds)
    # evalute_pred_dist_with_threshold(np.array(seg_t[70:125,:,:]),input_d[70:125,:,:],thresholds)
    #save_view(d1,seg_t)


    #d1,d2,t1 = main()
