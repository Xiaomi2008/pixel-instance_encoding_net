import os
import argparse
from experiment import experiment_config, experiment
def main():
	parser = argparse.ArgumentParser(description='main arguments')
	parser.add_argument('-a','--action', type=str, default='train', \
    	                 help='Option: train, predict, segment')
	parser.add_argument('-e','--exp_cfg',type =str, default='conf/experiment_1.toml', \
    	               help = 'path of the experiment_config file')
	args = parser.parse_args()
	exp_cfg = experiment_config(args.exp_cfg)
	exp_obj = experiment(exp_cfg)

	if args.action == 'train':
		exp_obj.train()
	elif args.action == 'predict':
		print ('predicting')
		exp_obj.predict()

if __name__ == '__main__':
    # configure which gpu or cpu to use
    #os.environ['CUDA_VISIBLE_DEVICES'] = '8'
    main()