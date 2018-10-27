# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import models
import random
import visdom
from utils.data_load import TrainDataset, ValiDataset, TestDataset
from utils.functions import val, test
from cfg import opt
import pdb
import pickle
import time
import pandas as pd

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

""" 0 加载数据 """
mode_name = opt.data_path.split('/')[-1][:-4] + '_' + str(opt.model).split('(')[0].split('=')[-1].strip() #获取数据集名字和模型名字
with open(opt.data_path, 'rb') as f_data:
    train_imgs_paths, train_label, vali_imgs_paths, vali_label = pickle.load(f_data)

train_data = TrainDataset(train_imgs_paths, train_label, transform=opt.transform)
train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=opt.batch_size, num_workers=opt.n_works)
vali_data = ValiDataset( vali_imgs_paths, vali_label, transform=opt.transform)
vali_dataloader = DataLoader(dataset=vali_data, shuffle=True, batch_size=opt.batch_size, num_workers=opt.n_works)
""" 1 加载网络模型 """
model = opt.model
model.cuda()

""" 2 定义loss和优化器 """
criterion = nn.CrossEntropyLoss()
lr = opt.init_lr
optimizer = optim.Adam(model.parameters(), lr=lr)

""" 3 可视化环境初始化 """
vis = visdom.Visdom(env=mode_name, use_incoming_socket=False)

""" 4 训练 """
if __name__ == '__main__':
    loss_sum = 0
    j = 0
    best_acc = 0
    for epoch in range(opt.epochs):

        """满足条件，则开始训练预训练的模型"""
        if epoch == opt.n_start:
            for param in model.parameters():
                param.require_grad = True

        time_s = time.time()
        model.train()
        for (data_x, label) in train_dataloader:

            input = data_x.cuda()
            label = label.cuda()
            output = model(input)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            '''可视化训练loss'''
            j += 1
            loss_sum += loss.item()
            if j % opt.print_every == 0:
                loss_mean = loss_sum / opt.print_every
                loss_sum = 0
                print('--第{}epoch--第{}batch--train_loss:{}--'.format(epoch, j, loss_mean))
                vis.line(X=torch.Tensor([j]), Y=torch.Tensor([loss_mean]), win='train loss',
                         update='append' if j != opt.print_every else None,
                         opts=dict(title='train_loss', x_label='batch', y_label='loss'))

        '''可视化模型在验证集上的准确率'''
        acc_vali = val(model, vali_dataloader)
        time_e = time.time()
        print('--------------第{}epoch结束, acc_vali :{}, 耗时:{}min--------------'.format((epoch), acc_vali, (time_e-time_s)/60))
        vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([acc_vali]), win='validation accuracy',
                 update='append' if epoch != 0 else None,
                 opts=dict(title='vali_acc', x_label='epoch', y_label='accuracy'))

        """每个epoch后，检测验证集分数是否上升，上升则保存模型至本地，否则加载best模型，降低学习率"""
        if acc_vali > best_acc:
            best_acc = acc_vali
            trained_model_path = './trained_models/' + mode_name + '_' + str(round(best_acc, 3))
            torch.save(model.state_dict(), trained_model_path)
            vis.save([mode_name])
        if (epoch % opt.decay_every) == 0 & (acc_vali < best_acc):
            model.load_state_dict(torch.load(trained_model_path))
            lr = lr * opt.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr)
