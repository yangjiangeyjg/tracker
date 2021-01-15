import torch
from torch import nn
from torch.nn import functional as F
from A_my_tracker.utils.anchors import generate_anchor_base,_enumerate_shifted_anchor
from A_my_tracker.utils.utils import loc2bbox
from torchvision.ops import nms
import numpy as np

'''
最复杂的部分还是RPN网络
首先是建议框生成的类，这部分包括许多处理，有意减少保留下来的框
就是从密密麻麻的anchor中选择生成rois的过程
只前向计算不反向传播。
'''

class ProposalCreator():
    def __init__(self,
                 mode,
                 nms_thresh=0.7, #nms阈值
                 n_train_pre_nms=12000, #nms预先的，就是选择进nms的
                 n_train_post_nms=512, #nms最后留的
                 n_test_pre_nms=12000, #训练和测试的时候区分一下
                 n_test_post_nms=512,
                 min_size=16 #最小的框，因为我们的下采样比例就是16
                 ):
        self.mode = mode
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        # 将RPN网络预测结果转化成建议框roi，根据初始框和调整
        roi = loc2bbox(anchor, loc) #N 4
        # 利用slice进行分割，防止建议框超出图像边缘
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[1])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[0])
        # 宽高的最小值不可以小于16
        min_size = self.min_size * scale
        # 计算高宽
        ws = roi[:, 2] - roi[:, 0]
        hs = roi[:, 3] - roi[:, 1]
        # 防止建议框过小
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        # 取出成绩最好的一些建议框
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]
        keep = nms(
            torch.from_numpy(roi).cuda(), 
            torch.from_numpy(score).cuda(), 
            self.nms_thresh
        ) #nms处理
        keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi #N 4


'''
区域建议网络
也就是RPN
'''
class RPN(nn.Module):
    def __init__(
            self, in_channels=256, mid_channels=256, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            mode = "training",
    ):
        super(RPN, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride  # 步长也就是压缩的倍数
        self.proposal_layer = ProposalCreator(mode)
        # 每一个网格上默认先验框的数量
        n_anchor = self.anchor_base.shape[0] #这里是9
        # 先进行一个3x3的卷积
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # 分类预测先验框内部是否包含物体
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 回归预测对先验框进行调整
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        #正太分布初始化参数
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape  #n是图片的数量
        # 对共享特征层进行一个3x3的卷积
        h = F.relu(self.conv1(x)) #还要做个激活
        # 回归预测
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4) #维度变化
        # 分类预测
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2) #维度变化
        # 进行softmax
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)
        # 生成先验框
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        rois = list()
        roi_indices = list()

        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32) #有多张图片
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        #返回建议回归 建议分类  生成的先验眶 batch序号 anchor
        return rpn_locs, rpn_scores, rois, roi_indices, anchor



def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
