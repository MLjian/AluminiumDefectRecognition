# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import torchvision.transforms as transforms
from utils.functions import FixedRotation
from models.inception_v4 import v4
from models.resnet34 import ResNet34
class DefaultConfig():

    """训练参数"""
    # model = v4(num_classes=12)
    model = ResNet34
    data_path = 'data/data_processed/data_1.pkl'
    batch_size = 32
    epochs = 32
    init_lr = 0.0001
    lr_decay = 0.3
    n_works = 0
    print_every = 30
    decay_every = 5
    n_start = 2
    transform = transforms.Compose([transforms.Resize((320, 320)),#关键部分
                                    transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomGrayscale(),
                                    # transforms.RandomRotation(20),
                                    FixedRotation([0, 90, 180, 270]),
                                    transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    """单模型预测参数"""
    test_model = v4(num_classes=12)
    best_model_path = './trained_models/data_1_SingleTask_0.9517543974675631pkl'
    test_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

opt = DefaultConfig()