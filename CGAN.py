# Copyright (C) 2021 #
# @Time    : 2023/3/2 11:56
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : CGAN.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes, load_digits
import pandas as pd
import numpy as np
import datasets
import multi_mode_dataset
import importlib
import torch.autograd
import wandb
importlib.reload(datasets)
importlib.reload(multi_mode_dataset)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import math
from torch.autograd import Variable



class Mydataset(Dataset):

    # Initialization
    def __init__(self, data, label1,label2, mode='2D'):
        self.data, self.label1, self.label2, self.mode = data, label1, label2, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label1[index, :], self.label2[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label1[:, index, :], self.label2[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]




class generator(nn.Module):

    def __init__(self,noise_dim,input_dim,mode_dim,hidden_dim = 16):
        super(generator, self).__init__()

        self.noise_dim = noise_dim
        self.input_dim = input_dim
        self.mode_dim = mode_dim
        self.hidden_dim = hidden_dim

        self.embedding_layer = nn.Embedding(self.mode_dim, 1)
        self.fc_x = nn.Sequential(
            nn.Linear(self.noise_dim + self.mode_dim, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, self.input_dim),
            nn.Sigmoid()
        )


    def forward(self,X,M):
        batch_size, _ = X.size()
        embed_M = self.embedding_layer(M)

        X_in = torch.cat((X,embed_M.squeeze()),dim=1)
        gen_x = self.fc_x(X_in)

        return gen_x

class discriminator(nn.Module):


    def __init__(self,input_dim,mode_dim):

        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.mode_dim = mode_dim
        self.embedding_layer = nn.Embedding(self.mode_dim,1)

        self.fc_d = nn.Sequential(
            nn.Linear(self.input_dim+self.mode_dim,32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32,16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,1)
        )

    def forward(self,X,M):

        embed_M = self.embedding_layer(M)
        X_in = torch.cat((X,embed_M.squeeze()),dim=1)
        validaty = self.fc_d(X_in)

        return validaty

class CGANmodel(RegressorMixin,BaseEstimator):

    def __init__(self,input_dim, noise_dim, mode_dim, batch_size=32, epoch = 10, lr_G = 0.0001, lr_D = 0.001, gpu = torch.device('cpu'),
                 seed = 1998, step_size=50, gamma=0.5):
        super(CGANmodel, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(1998)

        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.mode_dim = mode_dim
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.device = gpu
        self.seed = seed
        self.stepsize = step_size
        self.gamma = gamma
        self.n_critic_iter = 5
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.adversarial_loss = torch.nn.MSELoss()
        self.d_loss_hist = []
        self.g_loss_hist = []

        self.generator = generator(noise_dim, input_dim=input_dim, mode_dim=mode_dim).to(self.device)
        self.discriminator = discriminator(input_dim, mode_dim=mode_dim).to(self.device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G, betas=(0.5, 0.9))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D, betas=(0.5, 0.9))

    def fit(self, X, y_m):
        X = self.scaler_X.fit_transform(X)

        dataset = Mydataset(torch.tensor(X, dtype=torch.float32, device=self.device),
                            y_m,
                            torch.tensor(X, dtype=torch.float32, device=self.device), '2D')

        for e in range(self.epoch):
            self.d_loss_hist.append(0)
            self.g_loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size)

            for i, (batch_X, batch_y_m,_) in enumerate(data_loader):
                valid = Variable(torch.Tensor(batch_X.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.Tensor(batch_X.size(0), 1).fill_(0.0), requires_grad=False)

                # 训练generator

                self.optimizer_G.zero_grad()
                z = Variable(torch.Tensor(np.random.normal(0, 1, size=(batch_X.shape[0], self.noise_dim))))
                rand_m = Variable(torch.nn.functional.one_hot(torch.randint(0,self.mode_dim,size=(batch_X.shape[0],)),num_classes=3))


                gen_x = self.generator(z,rand_m)

                validity = self.discriminator(gen_x,rand_m)

                g_loss = self.adversarial_loss(validity,valid)
                self.g_loss_hist[-1] += g_loss.item()

                g_loss.backward()
                self.optimizer_G.step()

                # 训练discriminator
                self.optimizer_D.zero_grad()

                validity_real = self.discriminator(batch_X, batch_y_m)
                d_real_loss = self.adversarial_loss(validity_real,valid)

                validity_fake = self.discriminator(gen_x.detach(),rand_m)
                d_fake_loss = self.adversarial_loss(validity_fake,fake)

                d_loss = 0.5 * ( d_real_loss + d_fake_loss )
                self.d_loss_hist[-1] += d_loss.item()

                d_loss.backward()
                self.optimizer_D.step()
            print('Epoch: {:.0f}, d_loss:{:.6f}, g_loss:{:.6f}'.
                  format(e + 1, self.d_loss_hist[-1], self.g_loss_hist[-1]))
        print('Optimization finished')

    def generate_data(self,amount):

        self.generator.eval()
        with torch.no_grad():
            rand_m = torch.nn.functional.one_hot(torch.randint(0, self.mode_dim, dtype=int, size=(500,)), num_classes=3)
            noise = torch.rand(size=(amount,self.noise_dim))
            gen_x = self.generator(noise,rand_m)
        gen_x = self.scaler_X.inverse_transform(gen_x.detach())

        return gen_x


if __name__ == '__main__':

    name = 'numerical'
    Epoch = 2001
    # login wandb
    # wandb.init(
    #     project='MGAN',
    #     name='MGAN_{}'.format(Epoch),
    # config = {
    #     "dataset": name,
    #     'epoch:':Epoch
    # }
    # )


    X_train, y_train, m_train, X_test, y_test, m_test = multi_mode_dataset.datasets_m(name)
    print(X_train.shape)
    X_train = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
    X_test = np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)

    train_m = torch.nn.functional.one_hot(torch.Tensor(m_train).to(torch.int64),num_classes=3)


    _, x_dim = X_train.shape

    model = CGANmodel(input_dim=x_dim,mode_dim=3,noise_dim=8,batch_size=8,epoch=Epoch,lr_D=0.0001,lr_G=0.0001)
    model.fit(X_train,train_m)


    Amount = 500
    gen_x = model.generate_data(amount=Amount)
    # np.savetxt(fname='{}_gen_X_{}.txt'.format(name,int(Amount)),X=gen_x)
    # wandb.finish()
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    presult_train = pca.fit_transform(X=X_train)
    presult_test = pca.transform(X_test)
    presult_gen = pca.transform(gen_x)
    plt.figure()
    plt.scatter(presult_train[:,0],presult_train[:,1],c='r',label='Train')
    plt.scatter(presult_test[:, 0], presult_test[:, 1], c='b',label='Test')
    plt.scatter(presult_gen[:, 0], presult_gen[:, 1], c='orange',label='Gen')
    plt.legend()

    # plt.figure()
    # plt.scatter(X_train[:,0],X_train[:,1],c='r',label='y_Train')
    # plt.scatter(gen_x[:,0],gen_x[:,-1],c='b',label='gen_y')
    plt.legend()
    plt.show()