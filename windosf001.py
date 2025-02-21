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
import csv

setup_seed(3407)
###################
#gpu设置
###################
device = torch.device("cuda:1"  if torch.cuda.is_available() else "cpu")
# cuda = True if torch.cuda.is_available() else False
# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

###################
#数据集文件
###################
File_name = 'FD001'


###################
#获得训练集测试集和验证集
###################
if __name__ == '__main__':
    ###################
    # 训练
    ###################

    xh_num = 10#重复实验
    loss_ls = []
    Score = []

    for K in range(1):
        masking_ratio = 0.1
        loss_ratio = 0.1
        loss_sum = 0
        Score_sum = 0
        windows = 32
        windows_nam = 30

        # Load preprocessed data
        X_train = sio.loadmat(File_name + '/' + File_name + '_window_size_trainX_new'+'_'+str(windows_nam)+'.mat')
        X_train = X_train['trainX_new']
        # print(len(X_train))
        X_train = X_train.reshape(len(X_train), windows, 14)
        # print(X_train.shape)
        Y_train = sio.loadmat(File_name + '/' + File_name + '_window_size_trainY'+'_'+str(windows_nam)+'.mat')
        Y_train = Y_train['trainY']
        Y_train = Y_train.transpose()

        X_test = sio.loadmat(File_name + '/' + File_name + '_window_size_testX_new'+'_'+str(windows_nam)+'.mat')
        X_test = X_test['testX_new']
        X_test = X_test.reshape(len(X_test), windows, 14)
        Y_test = sio.loadmat(File_name + '/' + File_name + '_window_size_testY'+'_'+str(windows_nam)+'.mat')
        Y_test = Y_test['testY']
        Y_test = Y_test.transpose()

        X_train = Variable(torch.Tensor(X_train).float())
        Y_train = Variable(torch.Tensor(Y_train).float())
        X_test = Variable(torch.Tensor(X_test).float())
        Y_test = Variable(torch.Tensor(Y_test).float())

        for num in range(xh_num):
            setup_seed(3407 + num)

            # #Hyperparameters
            batch_size = 256
            dim_val = 64
            dim_attn = 64
            n_heads = 4
            n_encoder_layers = 4
            n_encoder_post_layers = 4

            n_encoder_again_layers = 2
            n_decoder_rul_layers = 1
            max_rul = 125
            lr = 0.001
            dec_seq_len = 4
            output_sequence_length = 1
            input_size = windows
            input_dim = 14
            epochs = 100
            patience = 15

            save_path = File_name  # 当前目录下

            # Dataloader
            train_dataset = TensorDataset(X_train, Y_train)
            train_data, eval_data = random_split(train_dataset,
                                                 [round(0.95 * X_train.shape[0]), round(0.05 * X_train.shape[0])],
                                                 generator=torch.Generator().manual_seed(2022))
            train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
            eval_loader = Data.DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=True)
            test_dataset = TensorDataset(X_test, Y_test)
            test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

            ###################
            # 定义模型
            ###################

            # model = FMAE(dim_val, dim_attn, input_dim, input_size, output_sequence_length, masking_ratio, dec_seq_len,
            #              n_encoder_post_layers, n_encoder_layers, n_encoder_again_layers,n_decoder_rul_layers, n_heads)#,dropout = 0.1)
            #线性回归
            model = Lr(input_dim, input_size)


            model.to(device)
            # model = model#使用gpu
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
            # optimizer.cuda()
            criterion = nn.MSELoss().to(device)

            # Training  and testing
            loss_list = []
            train_loss_list = []
            test_loss_list = []
            train_time = []
            test_time = []
            val_loss_list = []
            model_loss = 1000
            es = EarlyStopping(File_name + '/', patience)

            for epoch in range(1, epochs + 1):
                # training
                model.train()
                start1 = time.time()
                for i, (X, Y) in enumerate(train_loader):
                    X, Y = X.to(device), Y.to(device)
                    rul, u, u1 = model(X)
                    loss = loss_ratio * torch.sqrt(criterion(rul * max_rul, Y * max_rul)) + (1-loss_ratio) * torch.sqrt(criterion(u, u1))
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

                # evaling
                model.eval()  ##切换至评估模式，模型中的drop层batchnorm会关闭
                loss_val_list = []
                for j, (batch_x, batch_y) in enumerate(eval_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    pre, _, _ = model(batch_x)
                    pre[pre < 0] = 0
                    loss_val = torch.sqrt(criterion(pre * max_rul, batch_y * max_rul))
                    loss_val_list.append(loss_val.item())
                val_loss_eopch = np.mean(np.array(loss_val_list))
                val_loss_list.append(val_loss_eopch)
                print('eval_loss = ', val_loss_eopch.item())

                torch.manual_seed(3407)
                # torch.cuda.manual_seed_all(3407)
                prediction_list = []
                for d, (bat_x, bat_y) in enumerate(test_loader):
                    start2 = time.time()
                    bat_x, bat_y = bat_x.to(device), bat_y.to(device)
                    prediction, _, _ = model(bat_x)
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

                es(val_loss_eopch, model)
                # 达到早停止条件时，early_stop会被置为True
                if es.early_stop:
                    print("Early stopping")
                    break  # 跳出迭代，结束训练

            # plt.plot(list(range(1, epoch + 1)), train_loss_list, 'r*--')
            # plt.plot(list(range(1, epoch + 1)), test_loss_list, 'b*--')
            # plt.plot(list(range(1, epoch + 1)), val_loss_list, 'g*--')
            # plt.show()

            # testing
            model_1 = torch.load(File_name + '/' + 'lr.pth').to.device()
            model_1.eval()
            torch.manual_seed(3407)
            Y_test_numpy = Y_test.detach().numpy()  ##使其不具有梯度
            test_list = []

            for k, (batch_x, batch_y) in enumerate(test_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                prediction, _, _ = model_1(batch_x)
                prediction[prediction < 0] = 0
                test_list.append(prediction)
            test_all = torch.cat(test_list).detach().cpu().numpy()
            test_all_tensor = torch.from_numpy(test_all)
            test_loss_last = torch.sqrt(criterion(test_all_tensor * 125, Y_test * 125))
            test_score_last = myScore(Y_test_numpy * 125, test_all * 125)
            print('************************')
            print('test_loss = ', test_loss_last.item(),
                  'test_score = ', test_score_last)
            print('************************')

            #visualize(Y_test * 125, test_all_tensor * 125)

            loss_sum = loss_sum+test_loss_last.item()
            Score_sum = Score_sum + test_score_last

        print('************************')
        print('k: '+str(K)+' test_loss_last = ', round(loss_sum / xh_num, 2),
            'k: '+str(K)+' test_score_last = ', round(Score_sum / xh_num, 2))
        print('************************')

        loss_ls.append(round(loss_sum / xh_num, 2))
        Score.append(round(Score_sum / xh_num, 2))

    with open(File_name+'/' + 'lr.csv', 'w', newline='') as f:  # 保存
        csv_file = csv.writer(f)
        csv_file.writerow(['win-K',"RMSE", "Score"])
        for i in range(1,len(loss_ls)+1):
            csv_file.writerow([str(0.1*(2*i-1)),str(loss_ls[i-1]), str(Score[i-1])])
    f.close()
