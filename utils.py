import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import random


def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())    
    return torch.softmax(m,-1)##-1为softmax的维度

def Sensor_a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())    
    return torch.softmax(m,-1)

def time_step_a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())    
    return torch.softmax(m,-1)

#attention

def attention(Q, K, V):
    a = a_norm(Q, K) 
    return  torch.matmul(a,  V)

def Sensor_attention(Q, K, V):
    a = Sensor_a_norm(Q, K) 
    return  torch.matmul(a,  V) 

def time_step_attention(Q, K, V):
    a = time_step_a_norm(Q, K)  
    return  torch.matmul(a,  V)

#AttentionBlock#过全连接层得到qkv
class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    
    def forward(self, x, kv = None): 
        if(kv is None):
            return attention(self.query(x), self.key(x), self.value(x))        
        return attention(self.query(x), self.key(kv), self.value(kv))

class Sensor_AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(Sensor_AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val) 
        self.key = Key(dim_val, dim_attn) 
        self.query = Query(dim_val, dim_attn)
    
    def forward(self, x, kv = None): 
        if(kv is None):
            return Sensor_attention(self.query(x), self.key(x), self.value(x))        
        return Sensor_attention(self.query(x), self.key(kv), self.value(kv))
    
class time_step_AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(time_step_AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    
    def forward(self, x, kv = None):

        if(kv is None):
            return time_step_attention(self.query(x), self.key(x),   self.value(x))
        return time_step_attention(self.query(x), self.key(kv), self.value(kv))



# Multi-head self-attention 
class MultiHeadAttentionBlock(torch.nn.Module):  
    def __init__(self, dim_val, dim_attn, n_heads): 
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val,  dim_attn))       
        self.heads = nn.ModuleList(self.heads)
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
                      
        
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))           
        a = torch.stack(a, dim = -1) 
        a = a.flatten(start_dim = 2)        
        x = self.fc(a) 
        return x

class Sensor_MultiHeadAttentionBlock(torch.nn.Module): 
    def __init__(self, dim_val, dim_attn, n_heads): 
        super(Sensor_MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(Sensor_AttentionBlock(dim_val,  dim_attn))       
        self.heads = nn.ModuleList(self.heads)      
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
                              
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
        a = torch.stack(a, dim = -1) #在最后新建1维将a堆叠shape== (batch_size,sensor,dim_val_s,n_heads)
        a = a.flatten(start_dim = 2) #shape== (batch_size,sensor,dim_val_s*n_heads)
        x = self.fc(a) 
        return x

class TimeStepMultiHeadAttentionBlock(torch.nn.Module): 
    def __init__(self, dim_val, dim_attn, n_heads): 
        super(TimeStepMultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(time_step_AttentionBlock(dim_val,  dim_attn))        
        self.heads = nn.ModuleList(self.heads)        
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
                              
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))                    
        a = torch.stack(a, dim = -1) 
        a = a.flatten(start_dim = 2)         
        x = self.fc(a) 
        return x

# Query, Key and Value
class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val        
        self.fc1 = nn.Linear(dim_input, dim_val, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)       
        return x

class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)        
        return x

class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn        
        self.fc1 = torch.nn.Linear(dim_input, dim_attn, bias = False)
        
    def forward(self, x):

        x = self.fc1(x)      
        return x


#PositionalEncoding (from Transformer)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)#加1维变三维shape(512,1
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #shape(d_model/2)div_term = 1/10000的2i/d_model次方
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]. squeeze(1)
        #print(x.size())
        return x     

def visualize(true_rul, pred_rul):

    # # the true remaining useful life of the testing samples
    # true_rul = result.iloc[:, 0:1].to_numpy()
    # # the predicted remaining useful life of the testing samples
    # pred_rul = result.iloc[:, 1:].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.axvline(x=100, c='r', linestyle='--')
    plt.plot(true_rul, label='Actual Data')
    plt.plot(pred_rul, label='Predicted Data')
    plt.title('RUL Prediction on CMAPSS Data')
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")
    #plt.savefig('Transformer({}).png'.format(rmse))
    plt.show()

def visualize1(true_rul, pred_rul):

    # # the true remaining useful life of the testing samples
    # true_rul = result.iloc[:, 0:1].to_numpy()
    # # the predicted remaining useful life of the testing samples
    # pred_rul = result.iloc[:, 1:].to_numpy()

    true_rul, sort_index = torch.sort(true_rul,dim=0,  descending=True)
    pred_rul = pred_rul[sort_index].squeeze()

    plt.plot(true_rul, linewidth=1, color='blue')
    plt.plot(pred_rul, linewidth=1, color='orange')
    plt.title('RUL Prediction on CMAPSS Data-FD004')
    plt.legend()
    #plt.savefig('Transformer({}).png'.format(rmse))
    plt.xlabel("Engines unit number", fontsize=13)
    plt.ylabel("Remaining Useful Life", fontsize=13)

    # 修改坐标轴字体及大小
    plt.yticks(fontproperties='Times New Roman', size=13)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=13)

    plt.legend(['Actual RUL','Predicted RUL'])#添加图例
    plt.legend(['Actual RUL','Predicted RUL'],fontsize=13)#并且设置大小
    plt.show()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=20, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'mask2.pth')
        torch.save(model, path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


class mask_fea(nn.Module):
    def __init__(self, masking_ratio,input_size , dim_val,device):
        super(mask_fea, self).__init__()
        self.masking_ratio = masking_ratio
        self.mask_num = int((1-masking_ratio)*input_size*dim_val)
        self.fc_1 = torch.nn.Linear(self.mask_num, input_size * dim_val)
        self.devoce = device

    def forward(self,x):
        batch_num = x.shape[0]
        input_size = x.shape[1]
        dim_val = x.shape[2]
        post_fla = torch.reshape(x, (batch_num, -1))
        num_masked = input_size*dim_val-self.mask_num
        rand_indices = torch.rand(1, input_size * dim_val).argsort(axis=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        unmasked_indices, _ = torch.sort(unmasked_indices)
        masked_indices, _ = torch.sort(masked_indices)
        batch_range = torch.arange(batch_num)[:, None]
        mask = post_fla[batch_range, masked_indices]
        unmask = post_fla[batch_range, unmasked_indices]
        unmask = self.fc_1(unmask)
        unmask = torch.reshape(unmask, (batch_num, input_size, dim_val))
        #mask = torch.nn.Linear(mask.shape[1], input_size * dim_val)(mask)
        #mask = torch.reshape(mask, (batch_num, input_size, dim_val))
        return unmask, mask

def myScore(Target, Pred):
    tmp1 = 0
    tmp2 = 0
    for i in range(len(Target)):
        if Target[i] > Pred[i]:
            tmp1 = tmp1 + math.exp((-Pred[i] + Target[i]) / 13.0) - 1
        else:
            tmp2 = tmp2 + math.exp((Pred[i] - Target[i]) / 10.0) - 1
    tmp = tmp1 + tmp2
    return tmp

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    # np.random.seed(seed)  # numpy
    # random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
















 