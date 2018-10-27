# -*- coding: utf-8 -*-
"""
@brief : 用于训练stacking的二级分类器。主要包括如下操作：
            1）读取训练神经网络模型的数据，然后从中获取其验证集；
            2）经每个神经网络模型生成用于训练二级分类器的数据，并将其保存；
            3）利用神经网络模型生成的数据，训练二级分类器，并将训练好的二级分类器进行保存；
@interfaces :
@author : Jian
"""
import torch
from models import ResNet34, v4
import pickle
from utils.data_load import ValiDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

n_classes = 12
""" 1 加载验证集，用于训练lgb"""
data_path = './data/data_processed/data_1.pkl'
with open(data_path, 'rb') as f_data:
    _, _, vali_imgs_paths, vali_label = pickle.load(f_data)

""" 2 判断用于训练二级分类器的数据是否已经存在，若存在直接训练分类器，否则则要生成"""
clf_data_to_path = './ensemble/' + data_path.split('/')[-1].split('.')[0] + '_second.pkl'
clf_data_file = Path(clf_data_to_path)

if clf_data_file.exists():
    with open(clf_data_to_path, 'rb') as f:
        data_x, data_y = pickle.load(f)
else:
    """模型1"""
    model = v4(num_classes=n_classes)
    model.cuda()
    model_path = './trained_models/data_1_SingleTask_0.9517543974675631pkl'
    model.load_state_dict(torch.load(model_path))

    vali_data = ValiDataset(vali_imgs_paths, vali_label,
                             transform= transforms.Compose([
                                        transforms.Resize((400, 400)),
                                        transforms.CenterCrop(384),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    vali_dataloader = DataLoader(dataset=vali_data, shuffle=False, batch_size=12, num_workers=0)

    out_1 = np.expand_dims(np.zeros(n_classes), 0)
    label_1 = []
    for (x_batch,y_batch) in vali_dataloader:
        x_batch = x_batch.cuda()
        with torch.no_grad():
            o_batch = model(x_batch)

        out_1 = np.concatenate((out_1, o_batch.cpu().detach().numpy()), axis=0)
        label_1 = label_1 + y_batch.cpu().numpy().tolist()
    out_1 = out_1[1:, :]

    """模型2"""
    model = ResNet34
    model.cuda()
    model_path = './trained_models/data_1_ResNet_1'
    model.load_state_dict(torch.load(model_path))

    vali_data = ValiDataset(vali_imgs_paths, vali_label,
                            transform= transforms.Compose([
                                        transforms.Resize((400, 400)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    vali_dataloader = DataLoader(dataset=vali_data, shuffle=False, batch_size=12, num_workers=0)

    out_2 = np.expand_dims(np.zeros(n_classes), 0)
    label_2 = []
    for x_batch, y_batch in vali_dataloader:
        x_batch = x_batch.cuda()
        with torch.no_grad():
            o_batch = model(x_batch)

        out_2 = np.concatenate((out_2, o_batch.cpu().detach().numpy()), axis=0)
        label_2 = label_2 + y_batch.cpu().numpy().tolist()
    out_2 = out_2[1:, :]

    """拼接各个模型的结果"""
    if label_1 != label_2:
        raise ValueError("数据排列顺序有重大问题，请改代码！")
    data_x = np.concatenate((out_1, out_2), axis=1) # 修改处。。。。。。。。。。。。。。。。。。。。。。。。。
    data_y = np.array(label_1)
    data = (data_x, data_y)
    with open(clf_data_to_path, 'wb') as f_vali:
        pickle.dump(data, f_vali)
        print("用于训练二级分类器的数据已经生成，已保存至{}".format(clf_data_to_path))

""" 3 k折交叉验证，训练二级分类器 """

def clf_train(data_x, data_y, n_folds=80,  not_use_cv=True):
    """
    训练二级分类器，并将其保存至本地目录；
    Args:
        data_x: [n_samples, n_features] numpy数组
        data_y: [n_samples] numpy数组
        n_folds: 折数
        not_use_cv: True or False. 如果为True，则交叉验证调完参数后，再用所有数据训练一遍分类器。
    Returns: 无
    """
    print("用于训练二级分类器的样本共有{}个".format(len(data_y)))
    data_y = np.array(data_y)
    skf = StratifiedKFold(n_splits=n_folds)
    acc_sum = 0
    for train_idx, valid_idx in skf.split(data_x, data_y):
        train_x, valid_x = data_x[train_idx], data_x[valid_idx]
        train_y, valid_y = data_y[train_idx], data_y[valid_idx]
        clf = LogisticRegression(C=0.25)
        clf.fit(train_x, train_y)
        pred_valid = clf.predict(valid_x)
        acc_tmp = np.mean(np.equal(pred_valid, np.array(valid_y)).astype(np.float))
        acc_sum += acc_tmp
    acc_mean = acc_sum / n_folds
    print("---------{}折交叉验证后，二级分类器的准确率为{}-------".format(n_folds, acc_mean))

    """保存二级分类器模型"""
    if not_use_cv:
        clf = clf.fit(data_x, data_y)
    clf_to_path = clf_data_to_path[:-4] + '_clf_' + str(not_use_cv) + '.pkl'
    with open(clf_to_path, 'wb') as f_clf:
        pickle.dump(clf, f_clf)
    print("------已将二级分类器模型保存至本地目录{}--------".format(clf_to_path))

if __name__ == '__main__':

    """训练二级分类器"""
    clf_train(data_x, data_y, 80, not_use_cv=False)












