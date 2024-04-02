from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.RevIN import RevIN

from model.FLO import FLO

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from math import sqrt

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # x = x + self.dropout(new_x)
        # # y = x = self.norm1(x)
        # # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # # y = self.dropout(self.conv2(y).transpose(-1, 1))

        # return self.norm2(x )
# 
########################## origin ######################
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        out_list = []
        out_list.append(x)

        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask=attn_mask)
            out_list.append(x)


        #TODO add norm

        # if self.norm is not None:
            # x = self.norm(x)

        # return x, series_list, prior_list, sigma_list
        return  out_list

import torch.nn as nn

import torch.nn as nn

class CriticFunc(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super(CriticFunc, self).__init__()
        assert x_dim == y_dim
        d_model = x_dim 
        self.query_projection = nn.Linear(d_model,
                                          d_model)
        self.key_projection = nn.Linear(d_model,
                                        d_model)
        self.value_projection = nn.Linear(d_model,
                                          d_model)
        self.out_projection = nn.Linear(d_model, d_model)


    def forward(self, x, y):

        queriys = self.query_projection(x)

        keys = self.key_projection(y)

        # return re
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
        out = x.view(x.shape[0], self.input_dim)
        out = self._main(out)
        return out

class CriticFunc2_concat(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super(CriticFunc2_concat, self).__init__()

        #TODO feature dim 
        hidden = 128


        self.f3 = nn.Sequential(
            nn.Linear(x_dim+y_dim, hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )


        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


    def minmax(self, x):
        '''
            x: (b, n ,d)
        '''
        return torch.nn.functional.layer_norm(x, (x.shape[-1],))

    def forward(self, x, y, all=False):

        m = torch.concat([x,y], dim=-1)        

        m2 = self.f3(m)

        # if all == True:
            # re = torch.einsum('bnk,bmk->bnm',xx, yy)
        # else:
            # re = torch.einsum('bnk,bnk->bn',xx, yy)
        
        # re = torch.softmax(re, dim=-1)
        # print(m2.shape)

        return m2.squeeze(-1)

        return re.squeeze(-1)
    


class CriticFunc2_bilinear(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super(CriticFunc2_bilinear, self).__init__()

        #TODO feature dim 
        hidden = 128



        self.l0 = torch.nn.Linear(x_dim, hidden)
        self.l2 = torch.nn.Linear(y_dim, hidden)

        self.l1 = torch.nn.Linear(hidden, hidden)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def minmax(self, x):
        '''
            x: (b, n ,d)
        '''
        return torch.nn.functional.layer_norm(x, (x.shape[-1],))

    def forward(self, x, y, all=False):
        
        yy = self.minmax(self.l0(y))
        xx = self.minmax(self.l2(x))

        xx = self.l1(xx)
        xx = self.minmax(xx)

        if all == True:
            re = torch.einsum('bnk,bmk->bnm',xx, yy)
        else:
            re = torch.einsum('bnk,bnk->bn',xx, yy)
        
        return re

        return re.squeeze(-1)
    


class CriticFunc2(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super(CriticFunc2, self).__init__()

        #TODO feature dim 
        hidden = 128

        self.f1 = nn.Sequential(
            nn.Linear(y_dim, hidden//2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden),
        )

        self.f2 = nn.Sequential(
            nn.Linear(x_dim, hidden//2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden),
        )

        # self.f3 = 
        self.f3 = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )



        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


    def minmax(self, x):
        '''
            x: (b, n ,d)
        '''
        return torch.nn.functional.layer_norm(x, (x.shape[-1],))

    def forward(self, x, y, all=False):
        xx = self.f1(x)
        yy = self.f2(y)
        d = yy.shape[-1]

########################################################

        xx =  self.minmax(xx)
        yy = self.minmax(yy)

        if all == True:
            re = torch.einsum('bnk,bmk->bnm',xx, yy)
            return re

        re = torch.einsum('bnk,bnk->bn',xx, yy)
    



        # print(f"RE====> .  {torch.max(re)}, {torch.min(re)}")
        # re /= sqrt(d)
        # if self.mi_mode == 'test':
            # re = torch.softmax(re, dim=-1)  
        # re = torch.nn.functional.normalize(re,dim=-1)
########################################################

        # re = self.f3(torch.concat([xx,yy],dim=-1)).squeeze(-1)
        # re = torch.softmax(re, dim=-1)  
        # print(f"RE====> .  {torch.max(re)}, {torch.min(re)}")
########################################################

        
########################################################




        return re.squeeze(-1)
    
class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=256, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, method='infomax', critic='sep'):
        super(AnomalyTransformer, self).__init__()

        print(f'Create===>{d_model=}, {d_ff=}, {n_heads=}',)


        self.output_attention = output_attention

    
        # Encoding   1dconv + positionEmbedding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.method = method
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
        print(f'{critic=},==============>')

        if  critic == 'sep':
            self.critic_xz = CriticFunc2(d_model//2, d_model//2)
        elif critic == 'concat':
            self.critic_xz = CriticFunc2_concat(d_model//2, d_model//2)

        elif critic == 'bi':
            self.critic_xz = CriticFunc2_bilinear(d_model//2, d_model//2)
        
        else:
            raise f'Unavailable critic{critic}'

        self.u_func = deepcopy(self.critic_xz)
        # self.flo = FLO(d_model, d_model)
        self.ma_et = 0

        # self.k = torch.nn.Parameter(torch.tensor(0.))
        self.k = 0.1

        # self.b = torch.nn.Parameter(torch.tensor(0.))
        self.b = int(win_size * 0.1)
        self.tmp = torch.arange(0-win_size//2,win_size//2).cuda()
        self.disciminitor = MLP(d_model, 1, [d_model])

        # self.template = 

        t=1 
        
        
    def compute_reg(self, mlbo):

        
        # anchors = 1/(1+torch.exp( -self.k * self.tmp )).expand_as(mlbo)
        # loss = torch.nn.functional.mse_loss(mlbo, anchors)


        loss = torch.tensor(0.)

        # return loss - self.k 
        return loss 
        
        ...



    def compute_MLBO(self, x, z_q, loop=1):

        #TODO separate model and mi
        # x = x.detach()
        # z_q = z_q.detach()

        mlbo = []
        
        # loop = 3
        
        for i in range(loop):
            

            # batch_size, sequence_length, feature_dim = z_q.size()
            # all_permutations = torch.randperm(sequence_length*batch_size).cuda()
            # z_q_shuffle = torch.gather(z_q, 1, all_permutations.unsqueeze(-1).expand(-1, -1, feature_dim))
            # z_q_shuffle = z_q.reshape(-1, feature_dim)[all_permutations].reshape(batch_size, sequence_length, feature_dim)


            # torch
            batch_size, sequence_length, feature_dim = z_q.size()
            all_permutations = torch.stack([torch.randperm(sequence_length) for _ in range(batch_size)]).cuda()
            z_q_shuffle = torch.gather(z_q, 1, all_permutations.unsqueeze(-1).expand(-1, -1, feature_dim))


            # select min 
            # min_index = mlbo.detach().argmin(dim=-1)
            # batch_size, sequence_length, feature_dim = z_q.size() 
            # z_q_shuffle = torch.gather(z_q, 1, min_index.unsqueeze(-1).unsqueeze(-1).expand(-1, sequence_length, feature_dim))


            mlbo.append (self._compute_MLBO(x, z_q, z_q_shuffle))

        p = 0.5
        # mlbos = torch.quantile(mlbo, p, dim=-1)
        mlbo = torch.stack(mlbo, dim=-1)

        
        return torch.quantile(mlbo, p, dim=-1)
        # return torch.mean(mlbo, dim=-1)
        # return mlbo
        t = 1




    def _compute_MLBO(self, x, z_q, z_q_shuffle = None, 
                    #   method="jsd", 
                    #   method="infomax", 
                    # method='MINE',
                    #   method="MINE", 
                    #   method='flo',
                      ma_rate=0.001):
        

        method = self.method

        # assert method == 'MINE'

        if method == 'nwj':
            et = torch.exp((self.critic_xz(x, z_q_shuffle))-1)
            mlbo = self.critic_xz(x, z_q)  - (et)

        elif method == 'flo':

            # mlbo = self.flo(x, z_q)
            # g = self.l
            g = self.critic_xz(x, z_q)
            g0 = self.critic_xz(x, z_q_shuffle)
            u  = self.u_func(x, z_q)
            mlbo = u + torch.exp(-u+g0-g) - 1
            # mlbo = torch.nn.functional.softmax(mlbo, dim=-1)

        elif method == 'jsd':
            st = -torch.nn.functional.softplus(-self.critic_xz(x, z_q))
            ed = torch.nn.functional.softplus(self.critic_xz(x, z_q_shuffle))
            mlbo = st - ed

        elif method == "MINE":

            et = ((self.critic_xz(x, z_q_shuffle)))
            mlbo = self.critic_xz(x, z_q)  - (et)

            # et = (torch.exp(self.critic_xz(x, z_q_shuffle)))
            # self.ma_et = self.ma_et + ma_rate * (et.detach().mean() - self.ma_et)
            # mlbo = self.critic_xz(x, z_q)  - torch.log(et) * et.detach() / self.ma_et

            # et = ((self.critic_xz(x, z_q_shuffle)))
            # mlbo = self.critic_xz(x, z_q)  - (et)

        elif method == 'infomax':

            grim = self.critic_xz(x, z_q, all=True)
            A = torch.diagonal(grim, dim1=1, dim2=2)
            B = torch.log(grim.exp().sum(dim=-1))
            mlbo = A - B

        else:
            point = 1 / torch.exp(self.critic_xz(x, z_q_shuffle)).squeeze(dim=-1).mean()
            point = point.detach()

            if len(x.shape) == 3:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(
                    self.critic_xz(x, z_q_shuffle))  # + 1 + torch.log(point)
            else:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(
                    self.critic_xz(x, z_q_shuffle))


        # mlbo = torch.softmax(mlbo.reshape(-1), dim=-1).reshape(*mlbo.size())
        # mlbo = (mlbo.reshape(-1)).reshape(*mlbo.size())
        # mlbo = torch.sigmoid(mlbo)  
        return mlbo

    def get_intra_r(self, slow, rapid, permutation):

        res = []
        for s, r in zip(slow, rapid):
            r_shuffle = torch.gather(r, 1, permutation)
            res.append(torch.concat([s.detach(), r_shuffle], dim=-1))
            res[-1] = self.projection(res[-1])
        
        return res


    def get_intra_s(self, slow, rapid, permutation):

        res = []
        for s, r in zip(slow, rapid):
            s_shuffle = torch.gather(s, 1, permutation)
            res.append(torch.concat([s_shuffle, r.detach()], dim=-1))
            res[-1] = self.projection(res[-1])
        
        return res
        ...
    def get_inter_s(self, slow, rapid):

        res = []

        for id in range(len(slow)):
            tmp = slow[id]
            slow[id] = torch.concat([tmp[1:], tmp[0:1]])

        for s, r in zip(slow, rapid):
            res.append(torch.concat([s, r.detach()], dim=-1))
            res[-1] = self.projection(res[-1])

        return res

    def get_inter_r(self, slow, rapid):

        res = []

        for id in range(len(rapid)):
            tmp = rapid[id]
            rapid[id] = torch.concat([tmp[1:], tmp[0:1]])

        for s, r in zip(slow, rapid):
            res.append(torch.concat([s.detach(), r], dim=-1))
            res[-1] = self.projection(res[-1])

        return res   

    def forward(self, x, mode = 'train'):

        # print(x.shape)
        revin_layer = RevIN(num_features=x.shape[-1])
        # Instance Normalization Operation

        x = revin_layer(x, 'norm')

        enc_out = self.embedding(x) # 1dconv + positionEmbedding


        #TODO 
        enc_out = self.encoder(enc_out)
        
        f_dim = enc_out[-1].shape[-1]

        emb_rapid = []
        emb_slow = []



        for out in enc_out:
            emb_slow.append(out[:,:,:f_dim//2])
            emb_rapid.append(out[:,:,f_dim//2:])


        outputs = [self.projection(i) for i in enc_out]

        batch_size, sequence_length, feature_dim = emb_slow[-1].shape
        all_permutations_ = torch.stack([torch.randperm(sequence_length) for _ in range(batch_size)]).cuda()
        all_permutations = all_permutations_.unsqueeze(-1).expand(-1, -1, feature_dim)
        # z_q_shuffle = torch.gather(z_q, 1, all_permutations.unsqueeze(-1).expand(-1, -1, feature_dim))

        
        out_intra_r = self.get_intra_r(emb_slow, emb_rapid, all_permutations)
        out_intra_s = self.get_intra_s(emb_slow, emb_rapid, all_permutations)
        # emb_inter_r = self.get_inter_f(emb_slow, emb_rapid)
        out_inter_s = self.get_inter_s(emb_slow, emb_rapid)
        out_inter_r = self.get_inter_r(emb_slow, emb_rapid)


        return outputs, out_intra_s, out_intra_r, out_inter_s, out_inter_r, emb_slow, emb_rapid, all_permutations_

        # if self.output_attention:
            # return ret, enc_out
        # else:
            # return enc_out  # [B, L, D]
