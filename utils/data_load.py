# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pdb
from PIL import ImageFile


class TrainDataset(Dataset):
    def __init__(self, img_paths, label, transform=None):
        self.imgs_paths = img_paths
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.imgs_paths[index], self.label[index]
        img = Image.open(img_path).convert('RGB')


        #pdb.set_trace()
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs_paths)

class ValiDataset(Dataset):
    def __init__(self, img_paths, label, transform=None):
        self.imgs_paths = img_paths
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.imgs_paths[index], self.label[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs_paths)

class TestDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.imgs_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs_paths[index]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path

    def __len__(self):
        return len(self.imgs_paths)

