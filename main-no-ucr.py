import os
import time
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

def main(config, average):
    cudnn.benchmark = True
    print(config)
    # if (not os.path.exists(config.model_save_path)):
        # mkdir(config.model_save_path)

    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()


        print('============================  TRING END  ================================')

        # load_path = os.path.join('checkpoints',config.dataset, config.model_save_path+'checkpoint.pt')
        # print("Loading :", load_path)
        # solver.model.load_state_dict(torch.load(load_path))

        with torch.no_grad():
            # pr, re, f1score = solver.test()
            matrix = solver.test()
        
        matrix =  matrix + [config.id]

        if config.dataset == 'UCR' :
            import csv
            with open(f'checkpoints/UCR/{config.model_save_path}UCR.csv', 'a+') as f:
                writer = csv.writer(f)
                # writer.writerow([config.id, pr, re, f1score])
                average.append(matrix)
                writer.writerow(matrix)
                # average.append([config.id, pr, re, f1score])



    elif config.mode == 'test':

        solver.model.load_state_dict(torch.load( 'checkpoints/SMD/20231010134609checkpoint.pt'), strict=True)

        with torch.no_grad():
            solver.test()

    del solver
    return average
    # return solver



if __name__ == '__main__':

    wandb.login()
    parser = argparse.ArgumentParser()
    # parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument('--lr', type=float, default=0.0001)

    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument('--lr', type=float, default=0.0002)
    # parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--lr', type=float, default=0.00005)
    # parser.add_argument('--lr', type=float, default=0.00002)
    # parser.add_argument('--lr', type=float, default=0.00001)
    # parser.add_argument('--lr', type=float, default=  0.000005)
    # parser.add_argument('--lr', type=float, default=  0.000002)
    # parser.add_argument('--lr', type=float, default=  0.000001)
    # parser.add_argument('--lr', type=float, default=  0.000001)
    parser.add_argument('--lr', type=float, default=  0.0000005)

    parser.add_argument('--input_c', type=int, default=1)
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='UCR')
    parser.add_argument('--data_path', type=str, default='./dataset/kdd/dataset/UCR')
    parser.add_argument('--tem', type=float, default=5)
    parser.add_argument('--anomaly_ratio', type=float, default=0.5)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--id', type=int, default=-1)



    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_save_path', type=str, default='')
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--l_rec', type=float, default=0.)
    parser.add_argument('--l_intra_s', type=float, default=1.)
    parser.add_argument('--l_intra_r', type=float, default=1.)
    parser.add_argument('--l_mi', type=float, default=1.)
    parser.add_argument('--l_cons', type=float, default=1.)



    seed = np.random.randint(0,10000000)
    # seed = 2

    config = parser.parse_args()
    config.model_save_path =time.strftime("%Y%m%d%H%M%S")  + config.model_save_path
    config.seed = seed





    wandb.init(config=config)
    # wandb_config = wandb.config
    # args = wandb_config


    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    print("seed = ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark  = False
    os.environ['PYTHONHASHSEED'] = str(1)

    # main(args)

    results = []
    if config.dataset == 'UCR': 
        for id in range(1,250+1):
        # for id in range(226,250+1):
        # for id in range(78,80+1):
        # for id in range(1,5+1):
        # for id in [6,7,8]:
            print(f'================================= Start UCR {id} ======================================')
            config.id = id
            # print(config)
            try:
                result = main(config, results)

            except:
                print("Error in ", id)
    else:
        main(config)



    print("len results = ", len(results))
    results = np.average(results, axis=0)
    print("=============================  Results UCR ===================================")
    print(results)

    wandb.log({'f1score':results[3],
                'precision': results[1],
                'recall': results[2]})



