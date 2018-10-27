# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
num_classes = 12

ResNet34 = models.resnet34(pretrained=True)

for name, module in ResNet34._modules.items():
    print(name)
    print(module)





"""
for name, param in ResNet34.named_parameters():
    print(name)
"""
for param in ResNet34.parameters():
    param.require_grad = False

ResNet34 .fc = nn.Linear(512, num_classes)#512为resnet34倒数第二层神经元的个数