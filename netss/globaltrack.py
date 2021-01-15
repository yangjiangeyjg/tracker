

import torch
import numpy as np
import torch.nn as nn
from torchvision.ops import RoIAlign
from A_my_tracker.netss.resnet50 import resnet50
from A_my_tracker.netss.qg_rpn import RPN
from A_my_tracker.netss.qg_rcnn import RCNN
from A_my_tracker.netss.modulation import QG1

'''
搭建globaltrack的基本网络架构
给定x z 及相关处理的参数
注意z是已经根据初始框进行截取处理过的帧，至于z的获取不同数据格式和图像应该有不同的操作
x是图片帧
'''

class globaltrack(nn.Module):
    def __init__(self,num_classes, mode = "training",
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2),
                feat_stride = 16,
                anchor_scales = [8, 16, 32],
                ratios = [0.5, 1, 2]): #主要还是RPN的一些参数
        super(globaltrack, self).__init__()

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.feat_stride = feat_stride
        self.extractor, classifier = resnet50() #特征提取的backbone 和 RCNN处理
        #rpn和rcnn参数
        self.rpn=RPN(256, 256,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode = mode)
        self.rcnn=RCNN(n_class=num_classes + 1,roi_size=7,
                spatial_scale=(1. / self.feat_stride),
                classifier=classifier)
        self.QG1=QG1(c=256)

    # z是第一帧也就是模板帧 x是当前要检测的帧, 这里我们认为z已经截取好了
    def forward(self,x,z,scale=1.):
        img_size = x.shape[2:]
        #先进行特征提取，两流共享参数
        x_feature=self.extractor(x)
        z_feature=self.extractor(z)
        z_feature=RoIAlign(z_feature,(7,7),spatial_scale=1/16,sampling_ratio=-1)#这里注意
        #做相关计算,得到了特征图
        cor_feature=self.QG1.forward(x_feature,z_feature)
        #做RPN网络
        rpn_locs, rpn_scores, rois, roi_indices, anchor=\
            self.rpn.forward(cor_feature,img_size, scale)
        # 做RCNN网络，这里也需要特征图，第二次相关计算集成在rcnn模块里了，相关计算就在进入网络前
        rcnn_cls_locs, rcnn_scores = self.rcnn.forward(z_feature,cor_feature, rois, roi_indices)
        return rcnn_cls_locs, rcnn_scores, rois, roi_indices

    #检测BN模块
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

if __name__ =="__main__":
    model=globaltrack(1,"train")
    model = nn.DataParallel(model)
    model_path = 'E://py//pypy//A_my_tracker//model//fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
    model_path1 = 'E://py//pypy//A_faster-rcnn-pytorch-master/model_data//voc_weights_resnet.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(model)
    '''
    state_dict = torch.load(model_path1)
    #print(state_dict)
    model.load_state_dict(state_dict) #暴力加载参数
    #print(model_path2)
    #print(model)
    '''
    '''
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path1, map_location=device)
    #在自己定义上模型有的结构并且结构的参数的维度还得是相同的
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict  and np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    '''


'''
model_dict = model.state_dict()
print(model_dict)
#net = torch.load(model_path1, map_location=device)
# print(net)
#print(type(net))
#print(len(net))
'''

'''
import torch
pthfile = r'/home/bob/jj/pysot/snapshot_/checkpoint_e27.pth'
net = torch.load(pthfile, map_location='cpu')
net = dict(net)
with open('test.txt', 'a') as file0:
print(net, file=file0)
'''