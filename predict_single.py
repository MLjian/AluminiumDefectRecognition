# -*- coding: utf-8 -*-
"""
@brief : 利用单模型对测试集进行预测，并将结果保存至
@interfaces :
@author : Jian
"""
from cfg import opt
import torch
from utils.functions import test
from utils.data_load import TestDataset
from torch.utils.data import DataLoader
import pickle

"""加载模型"""
model = opt.test_model
model.cuda()
model.load_state_dict(torch.load(opt.best_model_path))

"""加载测试集"""
test_data_path = './data/data_processed/data_test.pkl'
with open(test_data_path, 'rb') as f_data:
    test_imgs_paths = pickle.load(f_data)

test_data = TestDataset(test_imgs_paths, transform=opt.test_transform)
test_dataloader = DataLoader(dataset=test_data, shuffle=True, batch_size=opt.batch_size, num_workers=opt.n_works)

"""预测，并保存结果"""
model_name = opt.best_model_path.split('/')[-1]
test(test_dataloader, model, model_name)

