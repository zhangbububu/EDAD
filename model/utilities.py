import os
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Set
from torch import nn  # type: ignor
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy import sparse as sp  # type: ignore
import torch  # type: ignore
from torch.utils import data  # type: ignore
from numpy.random import RandomState  # type: ignore
from typing import List, Tuple, Any, Optional
from scipy import sparse as sp  # type: ignore
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from livelossplot import PlotLosses

import time
import copy
import sys

class JSDGAN(nn.Module):
    def __init__(self,
                 critic: nn.Module, 
                 generator: Optional[nn.Module]=None, 
                 args: Optional[Dict] = None,
                 cuda: Optional[int] = None) -> None:
        '''
        Returns the ELBO of model
        MCMC only
        
        Input:
            critic          nn.Module    D(x) X->R
            generator       nn.Module    Z->X
            
        Output:
            JSD          float         variational objective
        '''
        
        super(JSDGAN,self).__init__()
        
        self.critic = critic
        self.generator = generator
        
    def forward(self,
               x_real: torch.Tensor, 
               x_fake: Optional[torch.Tensor]=None):
        
        n = x_real.shape[0]
        if x_fake is None:
            x_fake = self.generator(n=n)
        
        D_real = self.critic(x_real)
        D_fake = self.critic(x_fake)
        
        softplus = torch.nn.functional.softplus
        
        # V = -softplus(-D_real).mean()-softplus(D_fake).mean()
        V = (-softplus(-D_real)-softplus(D_fake)).mean()
        
        # Mutual information
        # I = D_real.mean()
        
        return V
    
    def MI(self, x_real: torch.Tensor):
        
        D_real = self.critic(x_real)
        I = D_real.mean()
        
        return I

    
    
    
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=[512,512], act_func=nn.ReLU()):
        super(MLP,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act_func = act_func
        
        layers = []
        for i in range(len(hidden_dim)):
            if i==0:
                layer = nn.Linear(input_dim, hidden_dim[i])
            else:
                layer = nn.Linear(hidden_dim[i-1], hidden_dim[i])
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            #layers.append(nn.ReLU(True))
            layers.append(act_func)
        if len(hidden_dim):                #if there is more than one hidden layer
            layer = nn.Linear(hidden_dim[-1], output_dim)
        else:
            layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        
        self._main = nn.Sequential(*layers)
        
    def forward(self, x):
        out = x
        # out = x.view(x.shape[0], self.input_dim)
        out = self._main(out)
        return out
            
                
            
            
class MITrainer(object):
    
    def __init__(self, 
                model: nn.Module):
        
        self.model = model
        
    def fit(self, 
            train_data,          # Tensor
            val_data  = None,    # Tensor
            args = None, 
            cuda = None):
        
        model = self.model
        
        since = time.time()
        liveloss = PlotLosses()

        num_epochs = 10
        batch_size = 128
        lr = 1e-4
        if args is not None:
            if args.get('num_epochs'): num_epochs = args['num_epochs']
            if args.get('batch_size'): batch_size = args['batch_size']
            if args.get('lr'): lr = args['lr']
                
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
#         val_loader = torch.utils.data.DataLoader(
#             val_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_loader = []
        
        data_loaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_loader), 'val': len(val_loader)}
        
        D = model.critic
        #G = model.generator
        
        opt_D = torch.optim.Adam(params=D.parameters(),lr=0.1*lr)
        #opt_G = torch.optim.Adam(params=G.parameters(),lr=lr)
#         optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        best_acc = -np.infty
        best_loss = np.infty
        best_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
#         criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):

            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_acc = torch.Tensor([0])
        #         running_regs = [0.]*len(regs)

                for i,(x,y) in enumerate(data_loaders[phase]):
                    
                    if phase == 'train': 
                        
                
                        opt_D.zero_grad()
                        
                        ###### Modified to train logit for MI ######
                        z_real = torch.cat([x,y],dim=1)
                        x0 = x[torch.randperm(x.size()[0])]
                        y0 = y[torch.randperm(y.size()[0])]
                        z_fake = torch.cat([x0,y0],dim=1)
                        loss = model(z_real,z_fake)
                        #############################################
                        
                        target_D = -loss
                        target_D.backward()
                        opt_D.step()
                        
#                         opt_G.zero_grad()
#                         loss = model(x)
#                         target_G = loss
#                         target_G.backward()
#                         opt_G.step()
                        

                    running_loss += target_D.item() 
            

                    print("\rIteration: {}/{}, Loss: {:.3f}\r".format(i+1, len(data_loaders[phase]), 
                                                        loss.item() ), end="")
                    sys.stdout.flush()

                epoch_loss = running_loss / dataset_sizes[phase]

        #         epoch_regs = [v/dataset_sizes[phase] for v in running_regs]
                avg_loss = epoch_loss
#                 t_acc = epoch_acc.numpy()[0]
        #             avg_regs = epoch_regs
        

            loss_dict = dict()
            loss_dict.update({'log loss': avg_loss})
    
        #     for j,(alph,reg,name,rnd_pair) in enumerate(regs):
        #         loss_dict.update({name: avg_regs[j], 'val_'+name: val_regs[j]})

            liveloss.update(loss_dict)

            liveloss.draw()
            
            #x_fake = model.generator(n=10).detach()
            #fig1 = plot_digits(x_fake)
    
            # deep copy the model
#             if val_loss < best_loss:
#                 # print('Updated')
#                 best_epoch = epoch+1
#                 best_loss = val_loss
#                 #best_model_wts = copy.deepcopy(model.state_dict())
#                 #torch.save(model.state_dict(), model_path)
#                 self.save()
          
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
#         self.load()
        
        return None
    
    def save(self, model_path="model.wts"):
        torch.save(self.model.state_dict(), model_path)
        
    def load(self, model_path="model.wts"):
        self.model.load_state_dict(torch.load(model_path))
        
        
        
        
        
        
class GaussianMixtureSample(object):
    
    def __init__(self, mu, sigma, alpha=None):
        
        m = len(mu)
        
        if alpha is None:
            alpha = np.array([1./m]*m)
        if len(mu.shape)==1:
            mu = mu.reshape([-1,1])
            sigma = sigma.reshape([-1,1])
            
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        
    def sample(self, n):
        
        mvn = multivariate_normal
        
        samples = []
        
        ncat = np.random.multinomial(n, self.alpha)
        for i,ni in enumerate(ncat):
            samples.append(mvn.rvs(self.mu[i],self.sigma[i],size=ni))
        
        samples = np.concatenate(samples)
        np.random.shuffle(samples)
            
        return samples
    
    
    
class InfoNCE(nn.Module):
    def __init__(self,
         critic: nn.Module, 
         args: Optional[Dict] = None,
         cuda: Optional[int] = None) -> None:
        
        super(InfoNCE,self).__init__()
        self.critic = critic
        
    def forward(self, x, y_list):
        
        output = self.PMI(x, y_list)
        
        return output.mean()
    
    def PMI(self, x, y_list):
        '''
        x:    n x p
        y_list:    length-K list of [n x d] K>=2
        '''
        
        log_f = self.critic
        
        log_fxy = []
        
        K = len(y_list)
        
        for y in y_list:
            xy = torch.cat([x,y],1)
            log_fxy.append(log_f(xy))
        log_fxy0 = log_fxy[0]   
        log_fxy = torch.cat(log_fxy,1)
#         print(log_fxy0.shape)
#         print(torch.logsumexp(log_fxy, dim=1, keepdim=True).shape)
        output = log_fxy0 - torch.logsumexp(log_fxy, dim=1, keepdim=True) + torch.log(torch.Tensor([K]))
#         log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
        
        return output

        
class InfoNCE(nn.Module):
    def __init__(self,
         critic: nn.Module, 
         args: Optional[Dict] = None,
         cuda: Optional[int] = None) -> None:
        
        super(InfoNCE,self).__init__()
        self.critic = critic
        
    def forward(self, x, y_list):
        
#         '''
#         x:    n x p
#         y_list:    length-K list of [n x d] K>=2
#         '''
        
#         log_f = self.critic
        
#         log_fxy = []
        
#         K = len(y_list)
        
#         for y in y_list:
#             xy = torch.cat([x,y],1)
#             log_fxy.append(log_f(xy))
#         log_fxy0 = log_fxy[0]   
#         log_fxy = torch.cat(log_fxy,1)
# #         print(log_fxy0.shape)
# #         print(torch.logsumexp(log_fxy, dim=1, keepdim=True).shape)
#         output = log_fxy0 - torch.logsumexp(log_fxy, dim=1, keepdim=True) + torch.log(torch.Tensor([K]))
# #         log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
        
        output = self.PMI(x, y_list)
        
        return output.mean()
    
    def PMI(self, x, y_list):
        '''
        x:    n x p
        y_list:    length-K list of [n x d] K>=2
        '''
        
        log_f = self.critic
        
        log_fxy = []
        
        K = len(y_list)
        
        for y in y_list:
            xy = torch.cat([x,y],1)
            log_fxy.append(log_f(xy))
        log_fxy0 = log_fxy[0]   
        log_fxy = torch.cat(log_fxy,1)
#         print(log_fxy0.shape)
#         print(torch.logsumexp(log_fxy, dim=1, keepdim=True).shape)
        output = log_fxy0 - torch.logsumexp(log_fxy, dim=1, keepdim=True) + torch.log(torch.Tensor([K]))
#         log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
        
        return output

        
class InfoNCETrainer(object):
    
    def __init__(self, model):
        
        self.model = model
        
    def fit(self, train_data, K=3, sampler = None, args=None):
        
        model = self.model
        
        since = time.time()
        liveloss = PlotLosses()

        num_epochs = 10
        batch_size = 128
        lr = 1e-4
        if args is not None:
            if args.get('num_epochs'): num_epochs = args['num_epochs']
            if args.get('batch_size'): batch_size = args['batch_size']
            if args.get('lr'): lr = args['lr']
                
        D = model.critic
        
        opt_D = torch.optim.Adam(params=D.parameters(),lr=lr)
        
        best_acc = -np.infty
        best_loss = np.infty
        best_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
#         val_loader = torch.utils.data.DataLoader(
#             val_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_loader = []
        
        data_loaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_loader), 'val': len(val_loader)}
        
        for epoch in range(num_epochs):
            if sampler is not None:
                train_data = sampler.sample(len(train_data))
                train_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=batch_size, 
                            shuffle=True, num_workers=1, drop_last=True)
                
                data_loaders = {'train': train_loader, 'val': val_loader}


            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_acc = torch.Tensor([0])
        #         running_regs = [0.]*len(regs)

                for i,(x,y) in enumerate(data_loaders[phase]):
                    
                    if phase == 'train': 
                        
                
                        opt_D.zero_grad()
                        
                        ###### Modified to train logit for MI ######
                        
                        y_list = [y]
                        for k in range(K-1):
                            y0 = y[torch.randperm(y.size()[0])]
                            y_list.append(y0)
                        
                        
                        fnice = model(x, y_list)
                        loss = -fnice
                        #############################################
                        
                        target_D = loss
                        target_D.backward()
                        opt_D.step()
                    
                        

                    running_loss += target_D.item() 
            

                    print("\rIteration: {}/{}, Loss: {:.3f}\r".format(i+1, len(data_loaders[phase]), 
                                                        loss.item() ), end="")
                    sys.stdout.flush()

                epoch_loss = running_loss / dataset_sizes[phase]

                avg_loss = epoch_loss
        

            loss_dict = dict()
            loss_dict.update({'log loss': avg_loss})
    
        #     for j,(alph,reg,name,rnd_pair) in enumerate(regs):
        #         loss_dict.update({name: avg_regs[j], 'val_'+name: val_regs[j]})

            liveloss.update(loss_dict)

            liveloss.draw()
            
          
        time_elapsed = time.time() - since
        
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
#         self.load()
        
        return None
    
    
    
    
class FenchelInfoNCE(nn.Module):
    def __init__(self,
         critic: nn.Module, 
         u_func: nn.Module,
         args: Optional[Dict] = None,
         cuda: Optional[int] = None) -> None:
        
        super(FenchelInfoNCE,self).__init__()
        self.critic = critic
        self.u_func = u_func
        
    def forward(self, x, y, y0):
        
        '''
        x:    n x p
        y:    n x d true
        y0:   n x d fake 
        '''
    
#         g = self.critic(x, y)
#         g0 = self.critic(x, y0)
#         u  = self.u_func(x, y)
#         output = u + torch.exp(-u+g0-g) - 1
        output  = self.PMI(x,y,y0)
        return output.mean()
    
    def MI(self, x, y, K=10):
        mi = 0
        for k in range(K):
            y0 = y[torch.randperm(y.size()[0])]
            mi += self.forward(x,y,y0)
            
        return -mi/K      
    def PMI(self, x, y, y0):
        '''
        x:    n x p
        y:    n x d true
        y0:   n x d fake 
        '''
    
        g = self.critic(x, y)
        g0 = self.critic(x, y0)
        u  = self.u_func(x, y)
        output = u + torch.exp(-u+g0-g) - 1
        return output       
    
    
class Wrapper(nn.Module):
    def __init__(self, func):
        super(Wrapper,self).__init__()
        self.func = func
#         self.dx = dx
#         self.dy = dy
        
    def forward(self, x, y):
        xy = torch.cat([x,y],dim=1)
        
        return self.func(xy)
    
    
    
class FenchelInfoNCETrainer(object):
    
    def __init__(self, model):
        
        self.model = model
        
    def fit(self, train_data, sampler=None, args=None):
        
        model = self.model
        
        since = time.time()
        liveloss = PlotLosses()

        num_epochs = 10
        batch_size = 128
        lr = 1e-4
        if args is not None:
            if args.get('num_epochs'): num_epochs = args['num_epochs']
            if args.get('batch_size'): batch_size = args['batch_size']
            if args.get('lr'): lr = args['lr']
        
        
        D = model.critic
        U = model.u_func

        #opt_D = torch.optim.Adam(params=D.parameters(),lr=lr)
        opt_D = torch.optim.Adam(params=[*D.parameters(),*U.parameters()],lr=lr)
        
        
        best_acc = -np.infty
        best_loss = np.infty
        best_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
#         val_loader = torch.utils.data.DataLoader(
#             val_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_loader = []
        
        data_loaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_loader), 'val': len(val_loader)}
    
        for epoch in range(num_epochs):
            if sampler is not None:
                train_data = sampler.sample(len(train_data))
                train_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=batch_size, 
                            shuffle=True, num_workers=1, drop_last=True)
                
                data_loaders = {'train': train_loader, 'val': val_loader}

            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_acc = torch.Tensor([0])
        #         running_regs = [0.]*len(regs)

                for i,(x,y) in enumerate(data_loaders[phase]):
                    
                    if phase == 'train': 
                        
                
                        opt_D.zero_grad()
                        
                        ###### Modified to train logit for MI ######
                        
                    
                        y0 = y[torch.randperm(y.size()[0])]

                        
                        
                        fnice = model(x,y,y0)
                        loss = fnice
                        #############################################
                        
                        target_D = loss
                        target_D.backward()
                        opt_D.step()
                    
                        

                    running_loss += target_D.item() 
            

                    print("\rIteration: {}/{}, Loss: {:.3f}\r".format(i+1, len(data_loaders[phase]), 
                                                        loss.item() ), end="")
                    sys.stdout.flush()
             
                epoch_loss = running_loss / dataset_sizes[phase]

                avg_loss = epoch_loss
        

            loss_dict = dict()
            loss_dict.update({'log loss': avg_loss})
    
        #     for j,(alph,reg,name,rnd_pair) in enumerate(regs):
        #         loss_dict.update({name: avg_regs[j], 'val_'+name: val_regs[j]})

            liveloss.update(loss_dict)

            liveloss.draw()
            
          
        time_elapsed = time.time() - since
        
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
#         self.load()
        
        return None
    
    
    
class Alpha(nn.Module):
    def __init__(self,
         critic: nn.Module, 
         u_func: nn.Module,
         alpha,
         args: Optional[Dict] = None,
         cuda: Optional[int] = None) -> None:
        
        super(Alpha,self).__init__()
        self.critic = critic
        self.u_func = u_func
        self.alpha = torch.Tensor([alpha])
        
    def forward(self, x, y_list):
    
        output = self.PMI(x, y_list)

        return output.mean()
    
    def PMI(self, x, y_list):
        log_f = self.critic
        u = self.u_func(x)
        log_fxy = []
        
        K = len(y_list)
        alpha = self.alpha
        
        for y in y_list:
            if alpha > 0:
                log_fxy.append(log_f(x,y)+torch.log(alpha)-torch.log(torch.Tensor([K])))
        if alpha<1:
            log_fxy.append(u+torch.log(1-alpha))
        if alpha>0:
            log_fxy1 = log_fxy[0]-torch.log(alpha)+torch.log(torch.Tensor([K]))
        else:
            log_fxy1 = log_f(x,y_list[0])
        log_fxy_list = log_fxy
        y0 = y[torch.randperm(y.size()[0])]   
        log_fxy0 = log_f(x,y0)
        log_fxy = torch.cat(log_fxy,1)
        term1 = log_fxy1 - torch.logsumexp(log_fxy, dim=1, keepdim=True)  
        if alpha>0:
            log_fxy_list[0] = log_fxy0+torch.log(alpha)-torch.log(torch.Tensor([K]))
            
        log_fxy_denom = torch.cat(log_fxy_list,1)
        term2 = torch.exp(log_fxy0 - torch.logsumexp(log_fxy_denom, dim=1, 
                            keepdim=True))
        
#         print(log_fxy0.shape)
#         print(torch.logsumexp(log_fxy, dim=1, keepdim=True).shape)
        output = 1 + term1 - term2
        
        return output
    
    
        

        
class AlphaTrainer(object):
    
    def __init__(self, model):
        
        self.model = model
        
    def fit(self, train_data, K=3, sampler=None, args=None):
        
        model = self.model
        
        since = time.time()
        liveloss = PlotLosses()

        num_epochs = 10
        batch_size = 128
        lr = 1e-4
        if args is not None:
            if args.get('num_epochs'): num_epochs = args['num_epochs']
            if args.get('batch_size'): batch_size = args['batch_size']
            if args.get('lr'): lr = args['lr']
                
        D = model.critic
        U = model.u_func
        opt_D = torch.optim.Adam(params=[*D.parameters(),*U.parameters()],lr=lr)
        
        best_acc = -np.infty
        best_loss = np.infty
        best_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
#         val_loader = torch.utils.data.DataLoader(
#             val_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_loader = []
        
        data_loaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_loader), 'val': len(val_loader)}
        
        for epoch in range(num_epochs):
            if sampler is not None:
                train_data = sampler.sample(len(train_data))
                train_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=batch_size, 
                            shuffle=True, num_workers=1, drop_last=True)
                
                data_loaders = {'train': train_loader, 'val': val_loader}

            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_acc = torch.Tensor([0])
        #         running_regs = [0.]*len(regs)

                for i,(x,y) in enumerate(data_loaders[phase]):
                    
                    if phase == 'train': 
                        
                
                        opt_D.zero_grad()
                        
                        ###### Modified to train logit for MI ######
                        
                        y_list = [y]
                        for k in range(K-1):
                            y0 = y[torch.randperm(y.size()[0])]
                            y_list.append(y0)
                        
                        
                        fnice = model(x, y_list)
                        loss = -fnice
                        #############################################
                        
                        target_D = loss
                        target_D.backward()
                        opt_D.step()
                    
                        

                    running_loss += target_D.item() 
            

                    print("\rIteration: {}/{}, Loss: {:.3f}\r".format(i+1, len(data_loaders[phase]), 
                                                        loss.item() ), end="")
                    sys.stdout.flush()

                epoch_loss = running_loss / dataset_sizes[phase]

                avg_loss = epoch_loss
        

            loss_dict = dict()
            loss_dict.update({'log loss': avg_loss})
    
        #     for j,(alph,reg,name,rnd_pair) in enumerate(regs):
        #         loss_dict.update({name: avg_regs[j], 'val_'+name: val_regs[j]})

            liveloss.update(loss_dict)

            liveloss.draw()
            
          
        time_elapsed = time.time() - since
        
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
#         self.load()
        
        return None
    
    
    
class NWJ(nn.Module):
    def __init__(self,
         critic: nn.Module, 
         args: Optional[Dict] = None,
         cuda: Optional[int] = None) -> None:
        
        super(NWJ,self).__init__()
        self.critic = critic
        
    def forward(self, x, y, y0):
        
        output = self.PMI(x,y,y0)
        
        return output.mean()
    
    def MI(self, x, y, K=10):
        mi = 0
        for k in range(K):
            y0 = y[torch.randperm(y.size()[0])]
            mi += self.forward(x,y,y0)
            
        return -mi/K  
    
    def PMI(self, x, y, y0):
        '''
        x:    n x p
        y:    n x d true
        y0:   n x d fake 
        '''
    
        g = self.critic(x, y)
        g0 = self.critic(x, y0)
    
        output = g - torch.exp(g0-1)
        
        return output
    
    
    
    
class NWJTrainer(object):
    
    def __init__(self, model):
        
        self.model = model
        
    def fit(self, train_data, sampler=None, args=None):
        
        model = self.model
        
        since = time.time()
        liveloss = PlotLosses()

        num_epochs = 10
        batch_size = 128
        lr = 1e-4
        if args is not None:
            if args.get('num_epochs'): num_epochs = args['num_epochs']
            if args.get('batch_size'): batch_size = args['batch_size']
            if args.get('lr'): lr = args['lr']
        
        
        D = model.critic
        

        opt_D = torch.optim.Adam(params=D.parameters(),lr=lr)
        #opt_D = torch.optim.Adam(params=[*D.parameters(),*U.parameters()],lr=lr)
        
        
        best_acc = -np.infty
        best_loss = np.infty
        best_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
#         val_loader = torch.utils.data.DataLoader(
#             val_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_loader = []
        
        data_loaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_loader), 'val': len(val_loader)}
        
        for epoch in range(num_epochs):
            if sampler is not None:
                train_data = sampler.sample(len(train_data))
                train_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=batch_size, 
                            shuffle=True, num_workers=1, drop_last=True)
                
                data_loaders = {'train': train_loader, 'val': val_loader}
            
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_acc = torch.Tensor([0])
        #         running_regs = [0.]*len(regs)

                for i,(x,y) in enumerate(data_loaders[phase]):
                    
                    if phase == 'train': 
                        
                
                        opt_D.zero_grad()
                        
                        ###### Modified to train logit for MI ######
                        
                    
                        y0 = y[torch.randperm(y.size()[0])]

                        
                        
                        fnice = model(x,y,y0)
                        loss = -fnice
                        #############################################
                        
                        target_D = loss
                        target_D.backward()
                        opt_D.step()
                    
                        

                    running_loss += target_D.item() 
            

                    print("\rIteration: {}/{}, Loss: {:.3f}\r".format(i+1, len(data_loaders[phase]), 
                                                        loss.item() ), end="")
                    sys.stdout.flush()

                epoch_loss = running_loss / dataset_sizes[phase]

                avg_loss = epoch_loss
        

            loss_dict = dict()
            loss_dict.update({'log loss': avg_loss})
    
        #     for j,(alph,reg,name,rnd_pair) in enumerate(regs):
        #         loss_dict.update({name: avg_regs[j], 'val_'+name: val_regs[j]})

            liveloss.update(loss_dict)

            liveloss.draw()
            
          
        time_elapsed = time.time() - since
        
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
#         self.load()
        
        return None
    
    
class TUBA(nn.Module):
    def __init__(self,
         critic: nn.Module, 
         u_func: nn.Module,
         args: Optional[Dict] = None,
         cuda: Optional[int] = None) -> None:
        
        super(TUBA,self).__init__()
        self.critic = critic
        self.u_func = u_func
        
    def forward(self, x, y, y0):
        
        
        output = self.PMI(x, y, y0)
        
        return output.mean()
    
    def MI(self, x, y, K=10):
        mi = 0
        for k in range(K):
            y0 = y[torch.randperm(y.size()[0])]
            mi += self.forward(x,y,y0)
            
        return -mi/K        
    def PMI(self, x, y, y0):
        '''
        x:    n x p
        y:    n x d true
        y0:   n x d fake 
        '''
    
        g = self.critic(x, y)
        g0 = self.critic(x, y0)
        u  = self.u_func(x)
        #output = u + torch.exp(-u+g0-g) - 1
        output = -g + u + torch.exp(-u+g0) - 1
        
        return output
    
    
    
class TUBATrainer(object):
    
    def __init__(self, model):
        
        self.model = model
        
    def fit(self, train_data, sampler = None, args=None):
        
        model = self.model
        
        since = time.time()
        liveloss = PlotLosses()

        num_epochs = 10
        batch_size = 128
        lr = 1e-4
        if args is not None:
            if args.get('num_epochs'): num_epochs = args['num_epochs']
            if args.get('batch_size'): batch_size = args['batch_size']
            if args.get('lr'): lr = args['lr']
        
        
        D = model.critic
        U = model.u_func

        #opt_D = torch.optim.Adam(params=D.parameters(),lr=lr)
        opt_D = torch.optim.Adam(params=[*D.parameters(),*U.parameters()],lr=lr)
        
        
        best_acc = -np.infty
        best_loss = np.infty
        best_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
#         val_loader = torch.utils.data.DataLoader(
#             val_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_loader = []
        
        data_loaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_loader), 'val': len(val_loader)}
        
        for epoch in range(num_epochs):
            if sampler is not None:
                train_data = sampler.sample(len(train_data))
                train_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=batch_size, 
                            shuffle=True, num_workers=1, drop_last=True)
                
                data_loaders = {'train': train_loader, 'val': val_loader}

            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_acc = torch.Tensor([0])
        #         running_regs = [0.]*len(regs)

                for i,(x,y) in enumerate(data_loaders[phase]):
                    
                    if phase == 'train': 
                        
                
                        opt_D.zero_grad()
                        
                        ###### Modified to train logit for MI ######
                        
                    
                        y0 = y[torch.randperm(y.size()[0])]

                        
                        
                        fnice = model(x,y,y0)
                        loss = fnice
                        #############################################
                        
                        target_D = loss
                        target_D.backward()
                        opt_D.step()
                    
                        

                    running_loss += target_D.item() 
            

                    print("\rIteration: {}/{}, Loss: {:.3f}\r".format(i+1, len(data_loaders[phase]), 
                                                        loss.item() ), end="")
                    sys.stdout.flush()

                epoch_loss = running_loss / dataset_sizes[phase]

                avg_loss = epoch_loss
        

            loss_dict = dict()
            loss_dict.update({'log loss': avg_loss})
    
        #     for j,(alph,reg,name,rnd_pair) in enumerate(regs):
        #         loss_dict.update({name: avg_regs[j], 'val_'+name: val_regs[j]})

            liveloss.update(loss_dict)

            liveloss.draw()
            
          
        time_elapsed = time.time() - since
        
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
#         self.load()
        
        return None
    
    
    
class ToySampler(object):
    def __init__(self, rho,p):
        mu = np.array([[0,0]])
        sigma = np.array([[[1,rho],[rho,1]]])
        self.GMS = GaussianMixtureSample(mu,sigma)
        self.p = p
    def sample(self, n):
        p = self.p
        N = n*p
        x = self.GMS.sample(N)
        x = torch.Tensor(x)
        x1 = x[(n*0):(n*(0+1)),0].reshape([-1,1])
        y1 = x[(n*0):(n*(0+1)),1].reshape([-1,1])
        for i in range(p-1):
            i = i + 1
            xi = x[(n*i):(n*(i+1)),0].reshape([-1,1])
            yi = x[(n*i):(n*(i+1)),1].reshape([-1,1])
            x1 = torch.cat((x1,xi),1)
            y1 = torch.cat((y1,yi),1)             
        train_data = list(zip(x1,y1))
        
        return train_data