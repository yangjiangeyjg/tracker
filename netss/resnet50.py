from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

'''
resnet50作为我们的网络结构的特征提取部分
不过还没有参数初始化 用coco的faster rcnn的还是imagenet的？
肯定是用faster rcnn的了，直接一起搞，faster rcnn用哪个是问题
'''

model_urls = {
'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}  #模型的链接字典，用于下载模型，下载的应该是imagenet预训练的

'''
Bottleneck类
定义bottleneck残差块
'''
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # 1*1
        self.bn1 = nn.BatchNorm2d(planes) #标准化均值为0方差为1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                    padding=1, bias=False)  # 3*3，1填充不改变大小
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  #1*1，输出扩大4倍
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x #x就是输入
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        #下采样，对x进行，确定是Conv Block还是Identity Block
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out+residual
        out = self.relu(out)
        return out


'''
ResNet类，1000类是imagenet
'''
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                    bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 在resnet50中分别是3 4 6 3个残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)  #平均池化
        self.fc = nn.Linear(512 * block.expansion, num_classes)  #全连接输出
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #如果是卷积层的话
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d): #如果是标准化的话
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #改变维度，方便后续进行了个全连接
        x = self.fc(x) #全连接
        return x




'''
resnet50函数,返回特征提取器 分类器部分
'''
def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3]) #调用Resnet函数，conv2_x到conv5_x分别是3 4 6 3
    # 获取特征提取部分
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # 获取分类部分
    classifier = list([model.layer4, model.avgpool]) #没有要最后的全连接，第四块和平均池化
    features = nn.Sequential(*features)  #容器包装好
    classifier = nn.Sequential(*classifier)
    return features,classifier    #返回特征提取和分类部分

'''
debug来看看。模型的加载、存储等，参数等
'''
if __name__ =="__main__":
    model = ResNet(Bottleneck, [3, 4, 6, 3])  # 调用Resnet函数，conv2_x到conv5_x分别是3 4 6 3
    # 获取特征提取部分
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # 获取分类部分
    classifier = list([model.layer4, model.avgpool])  # 没有要最后的全连接，第四块和平均池化
    features = nn.Sequential(*features)  # 容器包装好
    classifier = nn.Sequential(*classifier)
    print(model)


