import argparse
import os
import numpy as np
from experiment import experiment_config, experiment
from MaskNet_experiment import masknet_experiment, masknet_experiment_config
from eval_predict import predict_config, em_seg_eval
from cremi.io import CremiFile
from cremi import Annotations, Volume


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
        exp_obj = experiment(exp_cfg)
        exp_obj.train()
    elif args.action == 'evaluation':
        print('evaluation')
        exp_cfg = predict_config(args.exp_cfg)
        exp_obj = em_seg_eval(exp_cfg)
        arand, voi = exp_obj.eval()
        print(arand)
        print(voi)
    elif args.action == 'predict':
        print('predicting ...')
        exp_cfg = predict_config(args.exp_cfg)
        exp_obj = em_seg_eval(exp_cfg)
        seg3D_dict = exp_obj.predict()
        make_seg_submission(seg3D_dict)
        # print(arand)
        # print(voi)
    elif args.action == 'train_mask':
        exp_cfg = masknet_experiment_config(args.exp_cfg)
        exp_obj = masknet_experiment(exp_cfg)
        exp_obj.train()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '8'
    main()
