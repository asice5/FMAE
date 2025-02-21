import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class Time_step_EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1,dropout=0.1):
        super(Time_step_EncoderLayer, self).__init__()
        self.attn = TimeStepMultiHeadAttentionBlock(dim_val, dim_attn , n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
    
    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)                 
        a = self.dropout(self.fc1(F.elu(self.fc2(x))))
        x = self.norm2(x + a)          
        return x  


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        
    def forward(self, x, enc):
        a = self.attn1(x) 
        x = self.norm1(a + x)        
        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)
        a = self.dropout(self.fc1(F.elu(self.fc2(x))))
        x = self.norm3(x + a)
        return x

class FMAE(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_dim,input_size, out_seq_len, masking_ratio,dec_seq_len, n_encoder_post_layers,
                 n_encoder_layers,n_encoder_again_layers,n_decoder_rul_layers,n_heads, device,dropout=0.1):
        super(FMAE, self).__init__()

        # assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must in range (0, 1), but got {}.'.format(
        #     masking_ratio)
        self.masking_ratio = masking_ratio
        self.dec_seq_len = dec_seq_len

        # self.dec_seq_len = dec_seq_len
        self.dropout = nn.Dropout(dropout)

        # Initiate Time_step encoder
        self.time_encoder_post = []
        for i in range(n_encoder_post_layers):
            self.time_encoder_post.append(Time_step_EncoderLayer(dim_val, dim_attn, n_heads).to(device))

        self.time_encoder = []
        for i in range(n_encoder_layers):
            self.time_encoder.append(Time_step_EncoderLayer(dim_val, dim_attn, n_heads).to(device))

        self.time_encoder_again = []
        for i in range(n_encoder_again_layers):
            self.time_encoder_again.append(Time_step_EncoderLayer(dim_val, dim_attn, n_heads).to(device))

        self.decoder_rul = []
        for i in range(n_decoder_rul_layers):
            self.decoder_rul.append(DecoderLayer(dim_val, dim_attn, n_heads).to(device))

        self.pos_t = PositionalEncoding(dim_val).to(device)
        self.timestep_enc_input_fc = nn.Linear(input_dim, dim_val)
        self.rul_fc = nn.Linear(input_size * dim_val, out_seq_len)
        self.out1_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)
        self.mask_fea = mask_fea(masking_ratio,input_size,dim_val,device)
        self.norm1 = nn.LayerNorm(dim_val)

    def forward(self, x):

        post_ = self.pos_t(self.timestep_enc_input_fc(x))

        unmask, mask = self.mask_fea(post_)

        ##对原输入进行直接特征提取用于还原
        u = self.time_encoder_post[0](post_)  # ((batch_size,timestep,dim_val_t))##加post试一下
        for time_encoder_post in self.time_encoder_post[1:]:
            u = time_encoder_post(u)

        ##对mask后的进行特征提取
        o = self.time_encoder[0](unmask)  # ((batch_size,timestep,dim_val_t))##加post试一下
        for time_enc in self.time_encoder[1:]:
            o = time_enc(o)

        ##对mask后的特征进行提取用于还原
        u1 = self.time_encoder_again[0](u)  # ((batch_size,timestep,dim_val_t))##加post试一下
        for time_encoder_again in self.time_encoder_again[1:]:
            u1 = time_encoder_again(u1)

        # p = torch.cat((u, o), dim=1)  # ((batch_size,timestep+sensor,dim_val))
        # p = self.norm1(p)

        # decoder receive the output of feature fusion layer.
        d = self.decoder_rul[0](self.timestep_enc_input_fc(x[:, -self.dec_seq_len:]), u1)

        ##x[:,-4:]decoder的输入shape == (batch_size,4,64),p为encoder输出为多头的另一个输入

        x = torch.sigmoid(self.out1_fc(F.elu(d.flatten(start_dim=1))))

        return x,u,o


class FMAE_xr(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_dim,input_size, out_seq_len, dec_seq_len, n_encoder_post_layers,
                 n_decoder_rul_layers,n_heads, dropout=0.1):
        super(FMAE_xr, self).__init__()

        self.dec_seq_len = dec_seq_len

        # self.dec_seq_len = dec_seq_len
        self.dropout = nn.Dropout(dropout)

        # Initiate Time_step encoder
        self.time_encoder_post = []
        for i in range(n_encoder_post_layers):
            self.time_encoder_post.append(Time_step_EncoderLayer(dim_val, dim_attn, n_heads).cuda())

        self.decoder_rul = []
        for i in range(n_decoder_rul_layers):
            self.decoder_rul.append(DecoderLayer(dim_val, dim_attn, n_heads).cuda())

        self.pos_t = PositionalEncoding(dim_val).cuda()
        self.timestep_enc_input_fc = nn.Linear(input_dim, dim_val)
        self.out1_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)

    def forward(self, x):

        post_ = self.pos_t(self.timestep_enc_input_fc(x))

        ##对原输入进行直接特征提取用于还原
        u = self.time_encoder_post[0](post_)  # ((batch_size,timestep,dim_val_t))##加post试一下
        for time_encoder_post in self.time_encoder_post[1:]:
            u = time_encoder_post(u)

        # decoder receive the output of feature fusion layer.
        d = self.decoder_rul[0](self.timestep_enc_input_fc(x[:, -self.dec_seq_len:]), u)

        ##x[:,-4:]decoder的输入shape == (batch_size,4,64),p为encoder输出为多头的另一个输入

        x = torch.sigmoid(self.out1_fc(F.elu(d.flatten(start_dim=1))))

        return x


class Lr(torch.nn.Module):
    def __init__(self,input_dim, input_size):
        super(Lr, self).__init__()

        self.lr = nn.Linear(input_dim*input_size,1)
    def forward(self,x):
        y = self.lr(x)
        y = y.reshape(y.shape[0])
        return y
