# Copyright (C) 2021 #
# @Time    : 2023/2/21 11:46
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : WGAN-GP.py
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
import importlib
import torch.autograd

importlib.reload(datasets)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import math
from torch.autograd import Variable


class Mydataset(Dataset):

    # Initialization
    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]


class generator(nn.Module):

    def __init__(self, noise_dim, output_dim):
        super(generator, self).__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim

        # 网络构造

        self.G_FC_layer1 = nn.Linear(self.noise_dim, 32)
        self.G_FC_layer2 = nn.Linear(32, 32)
        # self.G_FC_layer3 = nn.Linear(32, 16)
        self.G_FC_layer4 = nn.Linear(32, output_dim)

    def forward(self, X):
        _h = nn.LeakyReLU(inplace=True)(self.G_FC_layer1(X))
        _h = nn.LeakyReLU(inplace=True)(self.G_FC_layer2(_h))
        # _h = nn.LeakyReLU()(self.G_FC_layer3(_h))
        output = nn.Sigmoid()(self.G_FC_layer4(_h))

        return output


class discriminator(nn.Module):
    def __init__(self, input_dim):
        super(discriminator, self).__init__()

        self.input_dim = input_dim

        self.D_FC_layer1 = nn.Linear(self.input_dim, 32)
        self.D_FC_layer2 = nn.Linear(32, 16)
        # self.D_FC_layer3 = nn.Linear(128, 32)
        self.D_FC_layer4 = nn.Linear(16, 1)

    def forward(self, X):
        _h = nn.LeakyReLU(inplace=True)(self.D_FC_layer1(X))
        _h = nn.LeakyReLU(inplace=True)(self.D_FC_layer2(_h))
        # _h = nn.ReLU()(self.D_FC_layer3(_h))
        validaty = self.D_FC_layer4(_h)

        return validaty
    

class WGANGPmodel(BaseEstimator,RegressorMixin):
    
    def __init__(self,input_dim, noise_dim, batch_size=32, epoch = 10, lr_G = 0.0001, lr_D = 0.001, gpu = torch.device('cpu'),
                 seed = 1998, step_size=50, gamma=0.5):
        super(WGANGPmodel, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(1998)
        self.input_dim = input_dim
        self.noise_dim = noise_dim
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

        self.d_loss_hist = []
        self.g_loss_hist = []
        self.generator = generator(noise_dim,output_dim=input_dim).to(self.device)
        self.discriminator = discriminator(input_dim).to(self.device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),lr=lr_G,betas=(0.5,0.9))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),lr=lr_D,betas=(0.5,0.9))


    def compute_gradient_penalty(self,real_X,gen_X):
        alpha = torch.Tensor(np.random.random((real_X.shape[0],1)))

        interpolates = (alpha * real_X + ((1-alpha) * gen_X)).requires_grad_(True)

        interpolates_result = self.discriminator(interpolates)

        fake = Variable(torch.Tensor(real_X.shape[0], 1).fill_(1.0), requires_grad=False)


        # fake for direction 用于给梯度取反方向
        gradients = torch.autograd.grad(outputs=interpolates_result,inputs=interpolates,
                                        grad_outputs=fake,create_graph=True,retain_graph=True,only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0),-1)

        gradient_penalty = ((gradients.norm(2,dim=1) - 1) ** 2).mean()

        return gradient_penalty


    def fit(self,X):
        X = self.scaler_X.fit_transform(X)

        dataset = Mydataset(torch.tensor(X, dtype=torch.float32, device=self.device),
                            torch.tensor(X, dtype=torch.float32, device=self.device), '2D')

        for e in range(self.epoch):
            self.d_loss_hist.append(0)
            self.g_loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size)

            w_d  = 0
            # 训练鉴别器

            for i,(batch_X, batch_y ) in enumerate(data_loader):

                self.optimizer_D.zero_grad()

                z = Variable(torch.Tensor(np.random.uniform(0,1,(batch_X.shape[0],self.noise_dim))))

                gen_x = self.generator(z)

                real_validity = self.discriminator(batch_X)
                fake_validity = self.discriminator(gen_x.detach())

                gradient_penalty = self.compute_gradient_penalty(batch_X.data,gen_x.data)

                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 0.5 * gradient_penalty

                wassertein_distance = torch.mean(real_validity) - torch.mean(fake_validity)

                w_d += wassertein_distance.item()
                self.d_loss_hist[-1] += d_loss.item()

                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                if i % self.n_critic_iter == 0:

                    # 训练生成器



                    gen_x = self.generator(z)

                    fake_validity = self.discriminator(gen_x)

                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()
                    self.g_loss_hist[-1] += g_loss.item()



            w_distance_all = w_d/i

            print('Epoch: {:.0f}, d_loss:{:.6f}, WD:{:.6f}, g_loss:{:.6f}'.
                  format(e + 1, self.d_loss_hist[-1],w_distance_all ,self.g_loss_hist[-1]))



    def generate_data(self, amount):

        self.generator.eval()
        with torch.no_grad():
            noise = torch.Tensor(np.random.uniform(0,1,(amount,self.noise_dim)))
            gen_x = self.generator(noise)
        gen_x = self.scaler_X.inverse_transform(gen_x.detach())

        return gen_x

if __name__ == '__main__':

    name = 'CO2'

    X_train, y_train, X_test, y_test = datasets.data_selection(name)
    print(X_train.shape)
    X_train = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
    X_test = np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)

    _, x_dim = X_train.shape

    model = WGANGPmodel(input_dim=x_dim,noise_dim=16,batch_size=8,epoch=2000,lr_D=0.0001,lr_G=0.0001)
    model.fit(X_train)

    # 106 131 160 211 271 316 391
    # 95 118 145 176 230 271
    Amount = 500
    gen_x = model.generate_data(amount=Amount)
    # np.savetxt(fname='{}_gen_X_{}.txt'.format(name,int(Amount)),X=gen_x)

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