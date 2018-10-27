# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import torch
from models import ResNet34, v4
import pickle
from utils.data_load import TestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import pdb
import os
import pandas as pd

n_classes = 12
""" 1 加载测试集"""
data_path = './data/data_processed/data_test.pkl'
with open(data_path, 'rb') as f_data:
    test_imgs_paths = pickle.load(f_data)

""" 2 用每个神经网络预测测试集"""

"""模型1"""
print("------模型1开始预测.......................")
model = v4(num_classes=n_classes)
model.cuda()
model_path = './trained_models/data_1_SingleTask_0.9517543974675631pkl'
model.load_state_dict(torch.load(model_path))

test_data = TestDataset(test_imgs_paths,
                         transform= transforms.Compose([
                                    transforms.Resize((400, 400)),
                                    transforms.CenterCrop(384),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
test_dataloader = DataLoader(dataset=test_data, shuffle=False, batch_size=12, num_workers=0)

out_1 = np.expand_dims(np.zeros(n_classes), 0)
paths_1 = ()
for (x_batch,y_batch) in test_dataloader:
    x_batch = x_batch.cuda()
    with torch.no_grad():
        o_batch = model(x_batch)

    out_1 = np.concatenate((out_1, o_batch.cpu().detach().numpy()), axis=0)
    paths_1 = paths_1 + y_batch
out_1 = out_1[1:, :]

"""模型2"""
print("------模型2开始预测.......................")
model = ResNet34
model.cuda()
model_path = './trained_models/data_1_ResNet_0.9275412133761815pkl'
model.load_state_dict(torch.load(model_path))

test_data = TestDataset(test_imgs_paths,
                         transform= transforms.Compose([
                                    transforms.Resize((400, 400)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
test_dataloader = DataLoader(dataset=test_data, shuffle=False, batch_size=12, num_workers=0)

out_2 = np.expand_dims(np.zeros(n_classes), 0)
paths_2  = ()
for (x_batch,y_batch) in test_dataloader:
    x_batch = x_batch.cuda()
    with torch.no_grad():
        o_batch = model(x_batch)

    out_2 = np.concatenate((out_2, o_batch.cpu().detach().numpy()), axis=0)
    paths_2 = paths_2 + y_batch
out_2 = out_2[1:, :]

"""拼接各个模型的结果"""
print("各个神经网络模型结果进行拼接.......................")
if paths_1 != paths_2:
    raise ValueError("数据排列顺序有重大问题，请改代码！")

data_x = np.concatenate((out_1, out_2), axis=1) # 修改处。。。。。。。。。。。。。。。。。。。。。。。。。
print("-----------神经网络模型已经完成预测！ ------------")

"""用二级分类器进行预测"""
clf_path = './ensemble/data_1_second_clf_False.pkl'          #修改处。。。。。。。。。。。。。。。。。。。。。。。。。。。。
with open(clf_path, 'rb') as f_clf:
    clf = pickle.load(f_clf)
    preds = clf.predict(data_x)

df_submission = pd.DataFrame()
df_submission['filename'] = [os.path.basename(path) for path in paths_1]
df_submission['label'] = list(preds)

def digit2label(digit):
    if digit == 0:
        return 'norm'
    else:
        return 'defect{}'.format(digit)
df_submission['label'] = df_submission['label'].apply(digit2label)
to_path = './results/ensemble/' + clf_path.split('/')[-1][:-4] + '_submission.csv'
df_submission.to_csv(to_path, header=None, index=False)
print("测试集预测完成，已保存至目录{}".format(to_path))


