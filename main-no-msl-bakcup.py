import os
import wandb
import argparse
import random

from torch.backends import cudnn
from utils.utils import *
from functools import partial
from solver import Solver
from torch.utils.tensorboard import SummaryWriter

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        # solver.model.load_state_dict(torch.load('./checkpoint.pt'))
        solver.model.load_state_dict(torch.load('./tmp_check.pt'))
        with torch.no_grad():
            solver.test()

    return solver



if __name__ == '__main__':
    #1234
    parser = argparse.ArgumentParser()
    # parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    # parser.add_argument('--input_c', type=int, default=55)
    # parser.add_argument('--output_c', type=int, default=55)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--ks', type=float, nargs='+',default=[1,1,1,1,1])

    parser.add_argument('--input_c', type=int, default=55)
    parser.add_argument('--output_c', type=int, default=55)
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--data_path', type=str, default='./dataset/MSL')


    # parser.add_argument('--input_c', type=int, default=25)
    # parser.add_argument('--output_c', type=int, default=25)
    # parser.add_argument('--dataset', type=str, default='PSM')
    # parser.add_argument('--data_path', type=str, default='./dataset/PSM')

    # parser.add_argument('--input_c', type=int, default=25)
    # parser.add_argument('--output_c', type=int, default=25)
    # parser.add_argument('--dataset', type=str, default='SMAP')
    # parser.add_argument('--data_path', type=str, default='./dataset/SMAP')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=1)
    parser.add_argument('--wandb', type=bool, default=True)

    config = parser.parse_args()

    config.l_rec =  1
    config.l_intra_s = 1
    config.l_intra_r = 1
    config.l_mi = 1

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    seed = np.random.randint(0,10000000)
    print("seed = ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


    main(config)

 
#TODO 加入AAE类似的方式来规范化critic， 2.继续测试flo的性能 3.调参   4.运行dc detector  5。向最终的metric种加入recloss后好像G了，打开separate试试

#  对比最原始的时候， 有无separate的区别
'''

--anormly_ratio
1
--num_epochs
3
--batch_size
256
--mode
train
--dataset
MSL
--data_path
dataset/MSL
--input_c
55
--output_c
55

'''