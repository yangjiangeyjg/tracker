
from __future__ import  absolute_import
import os
import time
from collections import namedtuple
from utils.utils import AnchorTargetCreator,ProposalTargetCreator #获取数据用
from torch.nn import functional as F
from torch import nn
import torch as torch

'''
关于整个网络的训练部分，首先要包括损失的计算
分类用的是交叉熵，回归用的是smooth1
'''
#smooth损失 sigma 和 in_weight是参数，x和t是框参数的向量
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float() #0或者1
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

#定位损失计算，预测框 真实框 及真实框的类别标签
def loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape)
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    if pred_loc.is_cuda:
        gt_loc = gt_loc.cuda()
        in_weight = in_weight.cuda()
    # smooth_l1损失函数
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # 进行标准化
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss


LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'rcnn_loc_loss',
                        'rcnn_cls_loss',
                        'total_loss'
                        ])
'''
数据的准备也很重要，在utils部分实现
'''

class globaltracker_trainer(nn.Module):
    def __init__(self, globaltrack, optimizer):
        super(globaltracker_trainer, self).__init__()
        self.globaltrack = globaltrack #网络结构
        self.rpn_sigma = 3  #损失计算的超参数
        self.rcnn_sigma = 1
        self.optimizer = optimizer #优化器
        self.anchor_target_creator = AnchorTargetCreator() #rpn生成数据来训练，20000个选2000
        self.proposal_target_creator = ProposalTargetCreator() #rcnn生成数据来训练，512个选128
        self.loc_normalize_mean = globaltrack.loc_normalize_mean #位置信息的均值
        self.loc_normalize_std = globaltrack.loc_normalize_std #位置信息的方差
    '''
    参数是批图像对数据 框 标签 尺度 模板
    计算损失的前向传播函数
    '''
    def forward(self, imgs, bboxes, labels, scale,temp):
        n = imgs.shape[0]
        '''
        这里需要改，最终是4
        '''
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.') #batchsize必须是1？
        _, _, H, W = imgs.shape
        img_size = (H, W)
        # 获取真实框和标签
        bbox = bboxes[0]
        label = labels[0]
        # 获取公用特征层，特征提取
        features = self.globaltrack.extractor(imgs)
        # 获取rpn的输出
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.globaltrack.rpn(features, img_size, scale)
         # 获取建议框的置信度和回归系数
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        #   建议框网络的loss
        # 先获取建议框网络应该有的预测结果
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox.cpu().numpy(),
            anchor,
            img_size) #先得到训练的数据
        gt_rpn_label = torch.Tensor(gt_rpn_label).long()
        gt_rpn_loc = torch.Tensor(gt_rpn_loc)
        # 计算建议框网络的loss值
        rpn_loc_loss = loc_loss(rpn_loc,
                                     gt_rpn_loc,
                                     gt_rpn_label.data,
                                     self.rpn_sigma) #回归损失
        if rpn_score.is_cuda:
            gt_rpn_label = gt_rpn_label.cuda()
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1) #分类损失

        #计算rcnn网络的loss
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            bbox.cpu().numpy(),
            label.cpu().numpy(),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.globaltrack.rcnn(
            temp,
            features,
            sample_roi,
            sample_roi_index)

        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)

        if roi_cls_loc.is_cuda:
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), torch.Tensor(gt_roi_label).long()].cuda()
        else:
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), torch.Tensor(gt_roi_label).long()]

        gt_roi_label = torch.Tensor(gt_roi_label).long() #类别标签
        gt_roi_loc = torch.Tensor(gt_roi_loc) #位置系数

        rcnn_loc_loss = loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.rcnn_sigma) #rcnn部分的回归损失

        if roi_score.is_cuda:
            gt_roi_label = gt_roi_label.cuda()

        rcnn_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label) #rcnn的分类损失，使用交叉熵，参数为置信度和标签

        losses = [rpn_loc_loss, rpn_cls_loss, rcnn_loc_loss, rcnn_cls_loss] #四部分损失
        losses = losses + [sum(losses)] #损失加在一起
        return LossTuple(*losses)

    '''
    执行一次参数更新 优化的过程
    '''
    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad() #梯度数据全部清零
        losses = self.forward(imgs, bboxes, labels, scale) #计算损失
        losses.total_loss.backward() #损失反向传播来计算损失的梯度
        self.optimizer.step() #参数优化更新
        return losses #返回损失

