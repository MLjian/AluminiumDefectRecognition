# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import torch
from collections import OrderedDict
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random

# 数据增强：在给定角度中随机进行旋转
def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])

class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)


def val(model, dataloader):
    """
    用于验证集的评测
    Args:
        model: 已训练的model
        dataloader: 训练集的dataloader
    Returns:
        acc_vali: 验证集的准确率
    """
    model.eval()
    acc_sum = 0
    for ii, (input, label) in enumerate(dataloader):
        val_input = input.cuda()
        val_label = label.cuda()

        with torch.no_grad():
            output = model(val_input)
        acc_batch = torch.mean(torch.eq(torch.max(output, 1)[1], val_label).float())
        acc_sum += acc_batch.item()

    acc_vali = acc_sum / (ii + 1)
    model.train()
    return acc_vali

def test(dataloader, model, model_name):
    csv_map = OrderedDict({'filename': [], 'probability': []})
    # switch to evaluate mode
    model.eval()
    for i, (images, filepath) in enumerate(tqdm(dataloader)):
        # bs, ncrops, c, h, w = images.size()
        filepath = [os.path.basename(i) for i in filepath]
        image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

        with torch.no_grad():
            y_pred = model(image_var.cuda())

            # get the index of the max log-probability
            smax = nn.Softmax(1)
            smax_out = smax(y_pred)
        csv_map['filename'].extend(filepath)
        for output in smax_out:
            prob = ';'.join([str(i) for i in output.data.tolist()])
            csv_map['probability'].append(prob)

    result = pd.DataFrame(csv_map)
    result['probability'] = result['probability'].map(lambda x: [float(i) for i in x.split(';')])

    # 转换成提交样例中的格式
    sub_filename, sub_label = [], []
    for index, row in result.iterrows():
        sub_filename.append(row['filename'])
        pred_label = np.argmax(row['probability'])
        if pred_label == 0:
            sub_label.append('norm')
        else:
            sub_label.append('defect%d' % pred_label)

    # 生成结果文件，保存在result文件夹中
    submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
    to_path = './results/single/' + model_name + '_submission.csv'
    submission.to_csv(to_path, header=None, index=False)
    return


