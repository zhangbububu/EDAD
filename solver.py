from copy import deepcopy
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from matplotlib import pyplot as plt
import tqdm
from utils.utils import *
from model.EDAD import EDAD
from data_factory.data_loader import get_loader_segment
from torch.utils.tensorboard import SummaryWriter
from metrics.metrics import *


def my_kl_loss(p, q):
    # p,q shape is (B, head, L, L)
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    # sum(res, dim=-1)也就是对于每一个time point，这个点对其他店的series和prior的分布的差异（kl散度）
    # mean( ，dim=1）也就是对head个kl散度去个平均值
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def myf1_score(gt, pr, adjust=True, modify=False):
    # get anomaly intervals
    gt_aug = np.concatenate([np.zeros(1), gt, np.zeros(1)]).astype(np.int32)
    gt_diff = gt_aug[1:] - gt_aug[:-1]

    begin = np.where(gt_diff == 1)[0]
    end = np.where(gt_diff == -1)[0]

    intervals = np.stack([begin, end], axis=1)

    pa = pr.copy()


        # point adjustment
    if True:
        if adjust:
            for s, e in intervals:
                interval = slice(s, e)
                if pa[interval].sum() > 0:
                    pa[interval] = 1

        # confusion matrix
        TP = (gt * pa).sum()
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()
        FN = (gt * (1 - pa)).sum()

        assert (TP + TN + FP + FN) == len(gt)

    # Compute p, r, f1.
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, num_losses=5, patience=6, path='none', verbose=False, delta=0, ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0 
        self.best_score = [None] * num_losses
        self.early_stop = False
        self.val_loss_min = [float('inf')] * num_losses
        self.delta = delta
        self.path = path

    def __call__(self, val_losses, model):
        

        all_ok = True

        print('++================++')
        for i, val_loss in enumerate(val_losses):
            score = val_loss
            if np.isnan(score):
                self.early_stop = True
                print("Early stopping for nan")
                return 

            print(f'[{self.best_score[i]=}]---[{val_loss=}], {self.best_score[i] is None or score < self.best_score[i] - self.delta}')
            if self.best_score[i] is None:
                self.best_score[i] = score
                # self.save_checkpdoint(val_loss, model, i)
            elif score > self.best_score[i] - self.delta:
                all_ok = False
        print('++================++')

        if all_ok == True:
            for i, val_loss in enumerate(val_losses):
                self.best_score[i] = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter for loss : {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                    self.early_stop = True
                    print(f"Early stopping for loss===>")

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased,Saving model to', self.path)
        torch.save(model.state_dict(), self.path)
        

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset,
                                               config = config)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset,
                                               config = config)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset,
                                               config = config)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset,
                                               config = config)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = EDAD(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c,
                                        d_model=self.d_model, d_ff=self.d_model, 
                                        n_heads=self.head, e_layers=3, method=self.method)
        self.teacher = deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.bank = torch.randn(self.batch_size, )
        self.plot_cnt = 0
        # log_dir = os.path.join('logs', self.dataset)
        # if not os.path.exists(log_dir):
            # os.mkdir(log_dir)
        # self.summary_writer = SummaryWriter(
        #     os.path.join(log_dir, time.strftime('%y%m%d%H%M%S_'+self.dataset, time.localtime(time.time())))
        # )
        # self.summary_writer.add_scalar('train/loss',1,1)
        # exit(1)

        if torch.cuda.is_available():
            self.model.cuda()
            self.teacher.cuda()


    def cons_loss(self, inputs, ) :

        outputs, emb_intra_s, emb_intra_r, emb_inter_s, emb_inter_r, emb_mi, emb_mi_rapid, permutations = self.model(inputs)
        w1 = outputs

        outputs, emb_intra_s, emb_intra_r, emb_inter_s, emb_inter_r, emb_mi, emb_mi_rapid, permutations = self.teacher(inputs)
        w2 = outputs

        cons_loss = torch.nn.functional.mse_loss(w1[-1], w2[-1])

        return cons_loss

        ...
    def vali(self, vali_loader):

        self.model.eval()
        rec_loss_his = []
        mi_loss_his = []
        loss_outputs = AverageMeter()
        loss_inter_s = AverageMeter()
        loss_inter_r = AverageMeter()
        loss_intra_s = AverageMeter()
        loss_intra_r = AverageMeter()
        loss_mi      = AverageMeter()
        loss_cons      = AverageMeter()

        for i, (input_data, labels) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            outputs, emb_intra_s, emb_intra_r, emb_inter_s, emb_inter_r, emb_mi, emb_mi_rapid, permutations = self.model(input)



            #rec
            rec_outputs = 0
            rec_intra_r = 0
            rec_intra_s = 0
            rec_inter_s = 0
            rec_inter_r = 0
            for id in range(len(outputs))[-1:]:
                # print(id)
                rec_outputs += self.criterion(outputs[id], input)*1/len(outputs)
                rec_intra_s += self.criterion(emb_intra_s[id], input)*1/len(outputs)
                shuffle_inputs = torch.gather(input, 1, permutations.unsqueeze(-1).expand(-1, -1, input.shape[-1]))
                rec_intra_r += self.criterion(emb_intra_r[id], shuffle_inputs)*1/len(outputs)
# 
                # rec_intra_r += -self.criterion(emb_intra_r[id], input)*1/len(outputs)
                # rec_inter_s += -self.criterion(emb_inter_s[id], input)*1/len(outputs)

                # rec_intra_r += self.criterion(emb_intra_r[id], input)*1/len(outputs)
                rec_inter_s += -self.criterion(emb_inter_s[id], outputs[id])*1/len(outputs)
                rec_inter_r += self.criterion(emb_inter_r[id], input)*1/len(outputs)


            # rec_intra_s = rec_outputs - rec_intra_s.detach()   # intra_s  small
            # rec_intra_r = rec_outputs - rec_intra_r.detach()   # intra_r  large
            # rec_inter_s = rec_outputs - rec_inter_s.detach()   # inter_s  large
            # rec_inter_r = rec_outputs - rec_inter_r.detach()   # inter_r  large


            




            #MI loss, reg loss
            ###########################################################
            mi_loss = 0
            reg_loss = 0
            for id in range(len(emb_mi))[-1:]:
            # for id in range(len(emb_mi)):
                # for id2 in range(len(emb_mi)):
                    # if id >= id2:
                        # continue
                if 1==1:
                    mlbo = self.model.compute_MLBO(emb_mi[0],emb_mi[id], loop=3)
                    # mlbo = self.model.compute_MLBO(emb_mi[id],emb_mi[id2], loop=3)
                    # mi = mlbo[0]
                    mi_loss += mlbo.mean()

                    min_val, _ = torch.min(mlbo, dim=1, keepdim=True)
                    max_val, _ = torch.max(mlbo, dim=1, keepdim=True)
                    mlbo = (mlbo - min_val) / (max_val - min_val + 1e-5)
                    mlbo, _ = mlbo.sort(dim=-1)
                    
                    
                    #regularization
                    reg = self.model.compute_reg(mlbo)
                    reg_loss += reg

                    # print(mlbo.shape)

            mi_loss /= len(emb_mi)-1
            reg_loss /= len(emb_mi)-1

            cons_loss = self.cons_loss(input) 

            loss_outputs.update(rec_outputs.detach().cpu().item())
            loss_inter_s.update(rec_inter_s.detach().cpu().item())
            loss_intra_s.update(rec_intra_s.detach().cpu().item())
            loss_intra_r.update(rec_intra_r.detach().cpu().item())
            loss_mi.update(mi_loss.detach().cpu().item())
            loss_cons.update(cons_loss.detach().cpu().item())


        val_losses = [loss_outputs.avg, loss_intra_s.avg, loss_intra_r.avg, 
                      -loss_mi.avg,
                      loss_cons.avg]

        # wandb.log({
        #         'Valid/loss_outputs':loss_outputs.avg
        #     })

        return val_losses




        return np.average(rec_loss_his), np.average(mi_loss_his)

    def one_step_train_vali(self, inputs, target, ):


        
        ...




    def train(self):
        mode = "train"
        print("======================TRAIN MODE======================")

        # time_now = time.time()
        # path = self.model_save_path
        # if not os.path.exists(path):
            # os.makedirs(path)

        # self.save_root = os.path.join('checkpoints',self.dataset,self.model_save_path+'checkpoint.pt')
        early_stopping = EarlyStopping(num_losses=5, patience=3, verbose=True, path=os.path.join('checkpoints',self.dataset,self.model_save_path+'checkpoint.pt'))

        train_steps = len(self.train_loader)

        # with torch.no_grad():
        #         self.model.critic_xz.mi_mode = 'test'
        #         self.test()

        for epoch in (range(self.num_epochs)):
            
            # with torch.no_grad():
                # self.test()
            rec_losss=[]
            reg_losss=[]
            mi_losss=[]
            loss_his = AverageMeter()
            loss_outputs = AverageMeter()
            loss_inter_s = AverageMeter()
            loss_intra_s = AverageMeter()
            loss_intra_r = AverageMeter()
            loss_mi      = AverageMeter()
            loss_cons      = AverageMeter()

            min_loss = []
            max_loss = []

            mi = 0

            epoch_time = time.time()
            self.model.train()
            self.teacher.train()
            for i, (input_data, labels) in enumerate(tqdm.tqdm(self.train_loader)):
                self.optimizer.zero_grad()


                for tm in range(2):


                # TODO  transformer 和  mlbo 分步训练
                    if tm == 0:
                        for p in self.model.parameters():
                            p.requires_grad = True
                        for p in self.model.critic_xz.parameters():
                            p.requires_grad = False
                    else:
                        for p in self.model.parameters():
                            p.requires_grad = False
                        for p in self.model.critic_xz.parameters():
                            p.requires_grad = True


                    input = input_data.float().to(self.device)
                    input_cons = input
                    outputs, emb_intra_s, emb_intra_r, emb_inter_s, emb_inter_r, emb_mi, emb_mi_rapid, permutations = self.model(input)



                    #rec
                    rec_outputs = 0
                    rec_intra_r = 0
                    rec_intra_s = 0
                    rec_inter_s = 0
                    rec_inter_r = 0
                    for id in range(len(outputs))[-1:]:
                        # print(id)
                        rec_outputs += self.criterion(outputs[id], input)*1/len(outputs)
                        rec_intra_s += self.criterion(emb_intra_s[id], input)*1/len(outputs)
                        shuffle_inputs = torch.gather(input, 1, permutations.unsqueeze(-1).expand(-1, -1, input.shape[-1]))
                        rec_intra_r += self.criterion(emb_intra_r[id], shuffle_inputs)*1/len(outputs)
# 
                        # rec_intra_r += -self.criterion(emb_intra_r[id], input)*1/len(outputs)
                        # rec_inter_s += -self.criterion(emb_inter_s[id], input)*1/len(outputs)

                        # rec_intra_r += self.criterion(emb_intra_r[id], input)*1/len(outputs)
                        rec_inter_s += -self.criterion(emb_inter_s[id], outputs[id])*1/len(outputs)
                        rec_inter_r += self.criterion(emb_inter_r[id], input)*1/len(outputs)


                    # rec_intra_s = rec_outputs - rec_intra_s.detach()   # intra_s  small
                    # rec_intra_r = rec_outputs - rec_intra_r.detach()   # intra_r  large
                    # rec_inter_s = rec_outputs - rec_inter_s.detach()   # inter_s  large
                    # rec_inter_r = rec_outputs - rec_inter_r.detach()   # inter_r  large


                    




                    #MI loss, reg loss
                    ###########################################################
                    mi_loss = 0
                    reg_loss = 0
                    for id in range(len(emb_mi))[-1:]:
                    # for id in range(len(emb_mi)):
                        # for id2 in range(len(emb_mi)):
                            # if id >= id2:
                                # continue
                        if 1==1:
                            mlbo = self.model.compute_MLBO(emb_mi[0],emb_mi[id], loop=3)
                            # mlbo = self.model.compute_MLBO(emb_mi[id],emb_mi[id2], loop=3)
                            # mi = mlbo[0]
                            mi_loss += mlbo.mean()

                            min_val, _ = torch.min(mlbo, dim=1, keepdim=True)
                            max_val, _ = torch.max(mlbo, dim=1, keepdim=True)
                            mlbo = (mlbo - min_val) / (max_val - min_val + 1e-5)
                            mlbo, _ = mlbo.sort(dim=-1)
                            max_loss.append(max_val.max())
                            min_loss.append(min_val.min())

                            #regularization
                            reg = self.model.compute_reg(mlbo)
                            reg_loss += reg

                            # print(mlbo.shape)

                    mi_loss /= len(emb_mi)-1
                    reg_loss /= len(emb_mi)-1


                    ##################consistency loss#############
                    cons_loss = self.cons_loss(input_cons) 

                    loss = self.l_rec*rec_outputs + \
                        0*rec_inter_s + \
                        self.l_intra_s*rec_intra_s + \
                        self.l_intra_r*rec_intra_r + \
                        -self.l_mi*mi_loss     + \
                        reg_loss    +\
                        self.l_cons * cons_loss

                    # loss = self.ks[0]*rec_outputs + \
                    #        self.ks[1]*rec_inter_s + \
                    #        self.ks[2]*rec_intra_s + \
                    #        self.ks[3]*rec_intra_r + \
                    #        self.ks[4]*mi_loss     + \
                    #        reg_loss
                    # print(f'{self.ks[0], self.ks[4]}')
                    # (rec_outputs -2 * mi_loss).backward(retain_graph=True)
                    # loss = rec_outputs + mi_loss + reg_loss

                    loss_his.update(loss.detach().cpu().item())
                    loss_outputs.update(rec_outputs.detach().cpu().item())
                    loss_inter_s.update(rec_inter_s.detach().cpu().item())
                    loss_intra_s.update(rec_intra_s.detach().cpu().item())
                    loss_intra_r.update(rec_intra_r.detach().cpu().item())
                    loss_mi.update(mi_loss.detach().cpu().item())
                    loss_cons.update(cons_loss.detach().cpu().item())

                    # wandb.log({
                    #     'Train/loss_outputs':loss_outputs.avg
                    # })

                    


                    if i % 100 == 0:
                        print(f'MI==> max={sum(max_loss)/len(max_loss)}, min={sum(min_loss)/len(min_loss)}')
                        print(f'{mode=}, {rec_outputs.item()}, {mi_loss.item()}, {reg_loss.item()}')
                        print(f'[train_avg_loss={loss_his.avg:.4f}], small{loss_outputs.avg=:.4f}, small{loss_cons.avg=:.4f}, small{loss_intra_s.avg=:.4f}, small{loss_intra_r.avg=:.4f}, {loss_mi.avg=:.4f}')
                        
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    ##########Update teacher params #########
                    alpha = 0.999
                    for mean_param, param in zip(self.teacher.parameters(), self.model.parameters()):
                        mean_param.data.mul_(alpha).add_(1 - alpha, param.data)


            with torch.no_grad():
                res = self.test()
                # self.summary_writer.add_scalar('Test/f1-score',res[2], epoch )

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

# 
            with torch.no_grad():
                val_losses = self.vali(self.test_loader)
            

            ######################teacher or model ? ########
            early_stopping([val_losses[0], val_losses[1],val_losses[2], val_losses[3], val_losses[4]], self.model)

            # return 
            if early_stopping.early_stop == True:
                return 

            # adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):

        
        self.model.eval()
        self.teacher.eval()

        #TODO adjust temp
        # temperature = 100000000000
        temperature = self.tem
        # temperature = 
        print("test,temperature", temperature)
        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []

        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            outputs, emb_intra_s, emb_intra_r, emb_inter_s, emb_inter_r, emb_mi, emb_mi_rapid, permutations = self.model(input)

            rec_outputs = 0
            rec_intra_r = 0
            rec_intra_s = 0
            rec_inter_s = 0

            for id in range(len(outputs))[-1:]:
                rec_outputs += (criterion(outputs[id], input)*1/len(outputs)).mean(dim=-1)

            rec_outputs_numpy = rec_outputs.detach().cpu().numpy()

            mlbos = 0
            for id in range(len(emb_mi))[-1:]:
                    mlbo = -self.model.compute_MLBO(emb_mi_rapid[0],emb_mi_rapid[id], loop=100)
                    mlbos += torch.nn.functional.softmax( mlbo * temperature, dim=-1)

            mlbos = mlbos.detach().cpu().numpy()
            # attens_energy.append(( rec_outputs_numpy*mlbos ))
            attens_energy.append((mlbos))
            
            

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

         # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            outputs, emb_intra_s, emb_intra_r, emb_inter_s, emb_inter_r, emb_mi, emb_mi_rapid, permutations = self.model(input)

            rec_outputs = 0
            rec_intra_r = 0
            rec_intra_s = 0
            rec_inter_s = 0

            for id in range(len(outputs))[-1:]:
                rec_outputs += (criterion(outputs[id], input)*1/len(outputs)).mean(dim=-1)

            rec_outputs_numpy = rec_outputs.detach().cpu().numpy()

            mlbos = 0
            for id in range(len(emb_mi))[-1:]:
                    mlbo = -self.model.compute_MLBO(emb_mi_rapid[0],emb_mi_rapid[id], loop=100)
                    mlbos += torch.nn.functional.softmax( mlbo * temperature, dim=-1)

            mlbos = mlbos.detach().cpu().numpy()
            # attens_energy.append(( rec_outputs_numpy*mlbos ))
            attens_energy.append((mlbos))


        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        print("Threshold :", thresh)


        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        # inputss = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            
            input = input_data.float().to(self.device)
            outputs, emb_intra_s, emb_intra_r, emb_inter_s, emb_inter_r, emb_mi, emb_mi_rapid, permutations = self.model(input)

            rec_outputs = 0
            rec_intra_r = 0
            rec_intra_s = 0
            rec_inter_s = 0

            for id in range(len(outputs))[-1:]:
                rec_outputs += (criterion(outputs[id], input)*1/len(outputs)).mean(dim=-1)

            rec_outputs_numpy = rec_outputs.detach().cpu().numpy()

            mlbos = 0
            for id in range(len(emb_mi))[-1:]:
                    mlbo = -self.model.compute_MLBO(emb_mi_rapid[0],emb_mi_rapid[id], loop=100)
                    mlbos += torch.nn.functional.softmax( mlbo * temperature, dim=-1)

            mlbos = mlbos.detach().cpu().numpy()
            # attens_energy.append(( rec_outputs_numpy*mlbos ))
            attens_energy.append((mlbos))
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

        plt.clf()
        # plt.plot(attens_energy[:1000])
        # plt.plot(inputss[:1000])
        # for index, i in enumerate(test_labels[:1000]):
            # if i == 1 :
                # print(index)
                # plt.axvline(index,color='red', alpha=0.5, linewidth=1) 
        # plt.savefig('tmp'+str(self.plot_cnt)+'.png')

        self.plot_cnt += 1
        test_energy = np.array(attens_energy)
        ##############
        # thresh = np.percentile(test_energy, 100 - self.anomaly_ratio)
        # print("My Threshold :", thresh)
        ##############
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        # st = 0
        # ed = 10
        # for index, i in enumerate(pred):
        #     if i == 0 and test_labels[index] == 1:
        #         st = max(0, index-200)
        #         ed = min(len(pred)-1, index+200)

        # plt.plot(attens_energy[st:ed])
        # for index, i in enumerate(test_labels[st:ed]):
        #     if i == 1 :
        #         # print(index)
        #         plt.axvline(index,color='red', alpha=0.5, linewidth=1) 
        #         plt.axhline(thresh)

        # print(f'{st=},{ed=}, saved tmp{self.plot_cnt}.pnh')
        # plt.savefig('tmp'+str(self.plot_cnt)+'.png') 



        gt = test_labels.astype(int)


        
        ############################# Additional Metric #############################
        # matrix = []
        # scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        # for key, value in scores_simple.items():
        #     matrix.append(value)
        #     print('{0:21} : {1:0.4f}'.format(key, value))


        # precision, recall, f1_score = myf1_score(gt, pred, adjust=True)
        # print("pred:   ", pred.shape)
        # print("gt:     ", gt.shape)

        # detection adjustment
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
        
            if anomaly_state:
                pred[i] = 1
        
        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)
        
        
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        


        if self.dataset == "UCR":
            return matrix

        return  precision, recall, f_score



'''
for i in range

'''