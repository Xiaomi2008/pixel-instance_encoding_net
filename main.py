import os
import argparse
from experiment import experiment_config, experiment
from MaskNet_experiment import masknet_experiment, masknet_experiment_config


def main():
    parser = argparse.ArgumentParser(description='main arguments')
    parser.add_argument('-a', '--action', type=str, default='train', \
                        help='Option: train, predict, segment')
    parser.add_argument('-e', '--exp_cfg', type=str, default='conf/experiment_1.toml',
                        help='path of the experiment_config file')
    args = parser.parse_args()

    if args.action == 'train2D':
        exp_cfg = experiment_config(args.exp_cfg)
        exp_obj = experiment(exp_cfg)
        exp_obj.train()
    elif args.action == 'predict2D':
        print ('predicting')
        exp_cfg = experiment_config(args.exp_cfg)
        exp_obj = experiment(exp_cfg)
        exp_obj.train()
        exp_obj.predict()
    elif args.action == 'Train_mask':
        exp_cfg = masknet_experiment_config(args.exp_cfg)
        exp_obj = masknet_experiment(exp_cfg)
        exp_obj.train()

if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '8'
    main()



# import os
# import argparse
# from experiment import experiment_config, experiment
# from MaskNet_experiment import  masknet_experiment, masknet_experiment_config
# from eval_predict import predict_config, em_seg_eval


# def main():
#     parser = argparse.ArgumentParser(description='main arguments')
#     parser.add_argument('-a', '--action', type=str, default='Train2D', \
#                         help='Option: train2D, train_mask, evaluation, predict')
#     parser.add_argument('-e', '--exp_cfg', type=str, default='conf/experiment_1.toml',
#                         help='path of the experiment_config file')
#     args = parser.parse_args()

#     if args.action == 'train2D':
#         exp_cfg = experiment_config(args.exp_cfg)
#         exp_obj = experiment(exp_cfg)
#         exp_obj.train()
#     elif args.action == 'evaluation':
#         print ('evaluation')
#         exp_cfg = predict_config(args.exp_cfg)
#         exp_obj = em_seg_eval(exp_cfg)
#         arand,voi = exp_obj.eval()
#         print(arand)
#         print(voi)
#     elif args.action == 'Train_mask':
#         exp_cfg = masknet_experiment_config(args.exp_cfg)
#         exp_obj = masknet_experiment(exp_cfg)
#         exp_obj.train()

# if __name__ == '__main__':
#     # configure which gpu or cpu to use
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '8'
#     main()
