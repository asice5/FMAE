# -*- coding: utf-8 -*-
"""
@author: HD

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.utils.data.dataloader as Data
from torch.utils.data import random_split
import os
import time
from torch.autograd import Variable
from utils import *
from Network import *
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


File_name = 'FD002'
windows = 62

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Myscore function

if __name__ == '__main__':

    # Load preprocessed data
    X_train = sio.loadmat(File_name+'/'+File_name+'_window_size_trainX_new.mat')
    X_train = X_train['trainX_new']
    #print(len(X_train))
    X_train = X_train.reshape(len(X_train), windows, 14)
    #print(X_train.shape)
    Y_train = sio.loadmat(File_name+'/'+File_name+'_window_size_trainY.mat')
    Y_train = Y_train['trainY']
    Y_train = Y_train.transpose()

    X_test = sio.loadmat(File_name+'/'+File_name+'_window_size_testX_new.mat')
    X_test = X_test['testX_new']
    X_test = X_test.reshape(len(X_test), windows, 14)
    Y_test = sio.loadmat(File_name+'/'+File_name+'_window_size_testY.mat')
    Y_test = Y_test['testY']
    Y_test = Y_test.transpose()
    
    X_train = Variable(torch.Tensor(X_train).float())
    Y_train = Variable(torch.Tensor(Y_train).float())
    X_test = Variable(torch.Tensor(X_test).float())
    Y_test = Variable(torch.Tensor(Y_test).float())

    # #Hyperparameters
    batch_size = 256
    dim_val = 64
    dim_attn = 64
    n_heads = 4
    n_encoder_layers = 2
    n_encoder_post_layers = 4
    n_decoder_rul_layers = 1

    max_rul = 125
    lr = 0.005
    dec_seq_len = 4
    output_sequence_length = 1
    input_size = windows
    input_dim = 14
    epochs = 100
    masking_ratio = 0.5

    loss_ls = []
    loss_sum =0
    Score = []
    Score_sum = 0

    num = 5

    save_path = File_name # 当前目录下


    #Dataloader
    train_dataset = TensorDataset(X_train,Y_train)
    #train_data,eval_data =random_split(train_dataset,[round(0.85*X_train.shape[0]),round(0.15*X_train.shape[0])],generator=torch.Generator().cuda.manual_seed(2022))
    train_loader = Data.DataLoader(dataset=train_dataset,batch_size = batch_size,shuffle=True)
    #eval_loader = Data.DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test,Y_test)
    test_loader = Data.DataLoader(dataset=test_dataset,batch_size = batch_size,shuffle=False)

    #重复试验
    for i in range(1,num+1):
        #setup_seed(3407+i)
        torch.cuda.manual_seed_all(3407+i)
        # Initialize model parameters
        model = FMAE(dim_val, dim_attn, input_dim, input_size, output_sequence_length, masking_ratio, dec_seq_len,n_encoder_post_layers,
                     n_encoder_layers,n_decoder_rul_layers, n_heads)
        model.cuda()
        #model = model#使用gpu
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0)
        #optimizer.cuda()
        criterion = nn.MSELoss().cuda()

        # Training  and testing
        loss_list = []
        train_loss_list = []
        test_loss_list = []
        train_time = []
        test_time = []
        model_loss = 1000

        for epoch in range(1,epochs+1):
            # training
            model.train()
            start1 = time.time()
            for i, (X, Y) in enumerate(train_loader):
                X, Y = X.type(FloatTensor), Y.type(FloatTensor)
                rul,u,u1 = model(X)
                loss = 0.6 * torch.sqrt(criterion(rul * max_rul, Y * max_rul)) + 0.4 * torch.sqrt(criterion(u, u1))
                #loss = torch.sqrt(criterion(rul * max_rul, Y * max_rul))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            end1 = time.time()
            train_time.append(end1 - start1)
            loss_eopch = np.mean(np.array(loss_list))
            train_loss_list.append(loss_eopch)
            print('epoch = ', epoch,
                  'train_loss = ', loss_eopch.item())

            # testing
            # #model.eval()  ##切换至评估模式，模型中的drop层batchnorm会关闭
            # loss_val_list = []
            # loss_val = 0
            # for j, (batch_x, batch_y) in enumerate(eval_loader):
            #     batch_x, batch_y = batch_x.type(FloatTensor), batch_y.type(FloatTensor)
            #     pre,_,_= model(batch_x)
            #     pre[pre < 0] = 0
            #     loss_val = torch.sqrt(criterion(pre * max_rul, batch_y * max_rul))
            #     loss_val_list.append(loss_val.item())
            # val_loss_eopch = np.mean(np.array(loss_val_list))
            # # out_batch_pre = torch.cat(loss_val_list).detach().numpy()
            # # prediction_tensor = torch.from_numpy(out_batch_pre)
            # # test_loss = torch.sqrt(criterion(prediction_tensor * 125, Y_test * 125))
            # # test_loss_list.append(test_loss)
            # # Y_test_numpy = Y_test.detach().numpy()
            # # test_score = myScore(Y_test_numpy * 125, out_batch_pre * 125)
            # print('eval_loss = ', val_loss_eopch.item())

            # testing
            model.eval()  ##切换至评估模式，模型中的drop层batchnorm会关闭
            #setup_seed(2022)
            torch.cuda.manual_seed(2022)
            # torch.cuda.manual_seed_all(3407)
            prediction_list = []
            for j, (batch_x, batch_y) in enumerate(test_loader):
                start2 = time.time()
                batch_x, batch_y = batch_x.type(FloatTensor), batch_y.type(FloatTensor)
                prediction,_,_ = model(batch_x)
                end2 = time.time()
                test_time.append(end2 - start2)
                prediction[prediction < 0] = 0
                prediction_list.append(prediction)

            out_batch_pre = torch.cat(prediction_list).detach().cpu().numpy()
            prediction_tensor = torch.from_numpy(out_batch_pre)
            test_loss = torch.sqrt(criterion(prediction_tensor * 125, Y_test * 125))
            test_loss_list.append(test_loss)
            Y_test_numpy = Y_test.detach().cpu().numpy()
            test_score = myScore(Y_test_numpy * 125, out_batch_pre * 125)
            print('test_loss = ', test_loss.item(),
                  'test_score = ', test_score)

            # early_stopping(val_loss_eopch, model)
            # # 达到早停止条件时，early_stop会被置为True
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break  # 跳出迭代，结束训练

            #Model save
            if epoch > 0:
                if test_loss.item() < model_loss:
                    model_loss = test_loss.item()
                    File_Path = File_name+'/'
                    if not os.path.exists(File_Path):
                        os.makedirs(File_Path)
                    torch.save(model, File_name+'/'+'best_network.pth2')

        plt.plot(list(range(1, epoch + 1)), train_loss_list, 'r*--')
        plt.plot(list(range(1, epoch + 1)), test_loss_list, 'b*--')
        plt.show()

        test_time_mean = np.mean(test_time)
        train_time_sum = np.sum(train_time)
        train_time_mean = np.mean(train_time)
        print('Test_time:', test_time_mean)
        print('Train_time:', train_time_sum)

        # testing
        model_1 = torch.load(File_name+'/'+'best_network.pth2').cuda()
        model_1.eval()
        #setup_seed(2022)
        torch.cuda.manual_seed(2022)
        # torch.cuda.manual_seed_all(3407)
        Y_test_numpy = Y_test.detach().numpy()  ##使其不具有梯度
        test_list = []

        for k, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.type(FloatTensor), batch_y.type(FloatTensor)
            prediction,_,_ = model_1(batch_x)
            prediction[prediction < 0] = 0
            test_list.append(prediction)
        test_all = torch.cat(test_list).detach().cpu().numpy()
        test_all_tensor = torch.from_numpy(test_all)
        test_loss_last = torch.sqrt(criterion(test_all_tensor * 125, Y_test * 125))
        test_score_last = myScore(Y_test_numpy * 125, test_all * 125)
        print('test_loss = ', test_loss_last.item(),
              'test_score = ', test_score_last)

        # true, indices = torch.sort(Y_test * 125, descending=True ,dim =0)
        # prerul = Y_test_numpy * 125
        # pre = prerul[indices].squeeze(1)
        ##可视化
        visualize(Y_test * 125, test_all_tensor * 125)
        loss_ls.append(test_loss_last.float())
        loss_sum = loss_sum+test_loss_last
        Score.append(test_score_last)
        Score_sum = Score_sum + test_score_last

    print('test_loss_last = ', round(loss_sum.item()/num,2),
          'test_score_last = ', round(Score_sum/num,2))
    print('loss', loss_ls,
          's ', Score)



