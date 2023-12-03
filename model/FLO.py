import torch.nn as nn
# %%
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


import sys
from model.utilities import *


class BilinearFenchelInfoNCEOne(nn.Module):
    def __init__(self,
         critic: nn.Module, 
         u_func: Optional[nn.Module] = None,
         K: Optional[int] = None,
         args: Optional[Dict] = None,
         cuda: Optional[int] = None) -> None:
        
        super(BilinearFenchelInfoNCEOne,self).__init__()
        self.critic = critic
        self.u_func = u_func
        self.K = K
    def forward(self, x, y, y0=None,K=None):
        
        '''
        x:    n x p
        y:    n x d true
        y0:   n x d fake 
        '''
        if K is None:
            K = self.K 
#         g = self.critic(x, y)
#         g0 = self.critic(x, y0)
#         u  = self.u_func(x, y)
#         output = u + torch.exp(-u+g0-g) - 1
        output  = self.PMI(x,y,y0,K)
        output = torch.clamp(output,-5,15)
        return output.mean()
    
    def MI(self, x, y, K=10):
        mi = 0
        for k in range(K):
            y0 = y[torch.randperm(y.size()[0])]
            mi += self.forward(x,y,y0)
            
        return -mi/K      
    def PMI(self, x, y, y0=None, K=None):
        '''
        x:    n x p
        y:    n x d true
        y0:   n x d fake 
        '''

        if self.u_func is not None:
            # two func mode
            u  = self.u_func(x, y)
            if K is not None:
            
                for k in range(K-1):

                    if k==0:
                        y0 = y0
                        g0 = self.critic(x, y0)
                    else:
                        y0 = y[torch.randperm(y.size()[0])]
                        g0 = torch.cat((g0,self.critic(x, y0)),1)

                g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
                output = u + torch.exp(-u+g0_logsumexp-g)/(K-1) - 1
            else:               
                
                g = self.critic(x, y)
                g0 = self.critic(x, y0)
               
                output = u + torch.exp(-u+g0-g) - 1
        else:
            # one func mode
            gu = self.critic(x,y)
            if isinstance(gu, tuple):
                hx,hy,u = gu
                similarity_matrix = hx @ hy.t()
                pos_mask = torch.eye(hx.size(0),dtype=torch.bool)
                g = similarity_matrix[pos_mask].view(hx.size(0),-1)
                g0 = similarity_matrix[~pos_mask].view(hx.size(0),-1)
                g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
                output = u + torch.exp(-u+g0_logsumexp-g)/(hx.size(0)-1) - 1

            else:      
                g, u = torch.chunk(self.critic(x,y),2,dim=1)
                if K is not None:

                    for k in range(K-1):

                        if k==0:
                            y0 = y0
                            g0,_ = torch.chunk(self.critic(x,y0),2,dim=1)
                        else:
                            y0 = y[torch.randperm(y.size()[0])]
                            g00,_ = torch.chunk(self.critic(x,y0),2,dim=1)
                            g0 = torch.cat((g0,g00),1)

                    g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
                    output = u + torch.exp(-u+g0_logsumexp-g)/(K-1) - 1
                else:    

                    g0, _ = torch.chunk(self.critic(x,y0),2,dim=1)
                    output = u + torch.exp(-u+g0-g) - 1
        return output

# %%
class BilinearCritic(nn.Module):
    
    '''
    encoder_x : dx -> feature_dim
    encoder_y : dy -> feature_dim
    u_func : 2*feature_dim -> 1
    '''
    
    def __init__(self,
                 encoder_x: nn.Module,
                 encoder_y: nn.Module,
                 u_func: nn.Module,
                 tau: Optional[float] = 1.):
        super(BilinearCritic,self).__init__()

        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.u_func = u_func
        self.tau = torch.nn.Parameter(torch.Tensor([tau]))
        
    
    def forward(self, x, y, tau=None):
        if tau is None:
            tau = self.tau
        tau = torch.sqrt(tau)
        hx = self.norm(self.encoder_x(x))
        hy = self.norm(self.encoder_y(y))
        u = self.u_func(hx,hy)
        
        return hx/tau, hy/tau, u  
    
    def norm(self,z):
        return torch.nn.functional.normalize(z,dim=1)
        #return z/torch.sqrt(torch.square(z).sum(1).view(-1,1))
        #return z

# %%
class Wrapper(nn.Module):
    def __init__(self, func):
        super(Wrapper,self).__init__()
        self.func = func
#         self.dx = dx
#         self.dy = dy
        
    def forward(self, x, y):
        xy = torch.cat([x,y],dim=-1)
        
        return self.func(xy)

# %%
# lam = 1
# p = args['dim_z']
# args = {}
# args['lr'] = 1e-3
# args['latent_dim'] = 100
# args['num_epochs'] = 50
# args["input_dim"] = 2*p
#args['batch_size']=50
# feature_dim = 512
# encoder_x = MLP(p, output_dim=feature_dim)
# encoder_y = MLP(p, output_dim=feature_dim)
# u_func = Wrapper(MLP(2*feature_dim,hidden_dim=[128]))
# critic = BilinearCritic(encoder_x, encoder_y, u_func)
#critic = Wrapper(MLP(args["input_dim"],2))
# u_func = Wrapper(MLP(args["input_dim"]))
# mi_model = BilinearFenchelInfoNCEOne(critic)
#mitrainer = FenchelInfoNCETrainer(mi_model)

class FLO(torch.nn.Module):
    def __init__(self, x_dim, y_dim):
        super(FLO, self).__init__()
        #   def __init__(self, input_dim, output_dim=1, hidden_dim=[512,512], act_func=nn.ReLU()):
        f_dim = 128
        ec_x = MLP(x_dim, output_dim=f_dim, hidden_dim=[f_dim, f_dim])
        ec_y = MLP(y_dim, output_dim=f_dim, hidden_dim=[f_dim, f_dim])
        u_func = Wrapper(MLP(2*f_dim,hidden_dim=[128]))
        critic = BilinearCritic(ec_x, ec_y, u_func)
        self.model = BilinearFenchelInfoNCEOne(critic)


    
    def forward(self, x, y):

        mlbo = self.model(x, y)
        return mlbo

        


# mi_model = mi_model.to(device)

# %%
# encoder_left = MLP(14*28, args['dim_z'])
# encoder_right = MLP(14*28, args['dim_z'])

# encoder_left = encoder_left.to(device)
# encoder_right = encoder_right.to(device)

# # %%

# optimizer = torch.optim.Adam([*encoder_left.parameters(),
#             *encoder_right.parameters(),*mi_model.parameters()],lr=1e-4)

# # %%
# # number of epochs to train the model
# n_epochs = 50  # suggest training between 20-50 epochs
# mi_model.train() # prep model for training

# for epoch in range(n_epochs):
#     # monitor training loss
#     train_loss = 0.0
#     train_mi = 0.0
#     ###################
#     # train the model #
#     ###################
# #     model.train()
#     mi_model.train()
#     for images_left, images_right in X_train_loader:
#         # load the (un)labeled data
        
#         # clear the gradients of all optimized variables
#         optimizer.zero_grad()
        
#         z_left = encoder_left(images_left)
#         z_right = encoder_right(images_right)
#         mi_loss = mi_model(z_left,z_right,None)
        
#         # forward pass: compute predicted outputs by passing inputs to the model
# #         output = model(data)
#         # calculate the loss
#         total_loss = mi_loss
#         # backward pass: compute gradient of the loss with respect to model parameters
#         total_loss.backward()
#         optimizer.step()
#         # perform a single optimization step (parameter update)
        
#         # update running training loss
#         # train_loss += loss.item()*data.size(0)
#         train_mi += -mi_loss.item()*images_left.size(0)
        
#     train_mi = train_mi/len(X_train_loader.dataset)
#     print('Epoch: {}\t MI: {:4f}'.format(epoch+1, train_mi))