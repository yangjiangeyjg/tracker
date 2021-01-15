from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

'''
本部分就是创建数据集，找出数据送入模型
pytorch中自带Dataset类和Dataloader类
要实现一个继承自Dataset的类，来实现数据集的创建
len和getitem这两个函数，前者为数据集大小，后者查找数据和标签，主要是路径配置
DataLoader是一个迭代器，来取得数据喂入模型，可以设置一些参数
'''

class globaltrackDataset(Dataset):
    def __init__(self,train_lines,shape=[600,600],batch_size=1):
        self.train_lines = train_lines
        self.train_batches = len(train_lines) #数据集的大小
        self.shape = shape #长和宽
        self.batch_size = batch_size #这里先设为1

    def __len__(self):
        return self.train_batches #数据集大小

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    '''
    采样数据并进行处理，分数据集处理
    '''
    def get_random_data(self,datasets,annotation_line):
        if datasets=='LaSOT':
            line = annotation_line.split()
            image = Image.open(
                'H://LaSOTTest//zip//' + line[0] +'.jpg')
            iw, ih = image.size
            h, w = self.shape
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
            image = image.resize((w, h), Image.BICUBIC)

    def __getitem__(self, ):
        img=1
        box=1
        label=1
        return img, box, label


# DataLoader中collate_fn使用
def globaltrack_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    return images, bboxes, labels

