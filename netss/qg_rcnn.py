import torch
import torch.nn as nn
from torch.nn import Module
from torchvision.ops import RoIAlign
from A_my_tracker.netss.modulation import QG2

'''
这部分应该对应于我们的rcnn模块，这部分也应该做成多分类的
因为我们有多实例查询损失
也就是把整个resnet50的全连接换成我们的全连接
利用了一部分卷积,第四部分，还有个池化
之后就换成自己的全连接了
'''

# spatial_scale 将输入坐标映射到box坐标的尺度因子
# roi_indices在此代码中是多余的，因为我们实现的是batch_size=1的网络，一个batch只会输入一张图象。如果多张图象的话就需要存储索引以找到对应图像的roi
class RCNN(nn.Module):
    def __init__(self,n_class,roi_size, spatial_scale,classifier):
        super(RCNN, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(2048, n_class * 4)
        self.score = nn.Linear(2048, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        self.roi_size = roi_size  #好像是7*7
        self.spatial_scale = spatial_scale  #好像是十六分之一
        self.roi = RoIAlign((self.roi_size, self.roi_size), self.spatial_scale,sampling_ratio=-1)

    def forward(self, temp,x, rois, roi_indices):
        roi_indices = torch.Tensor(roi_indices).float()
        rois = torch.Tensor(rois).float()
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1) #[N, 5] (index, x, y, h, w)
        xy_indices_and_rois = indices_and_rois[:, [0, 1, 2, 3, 4]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        # 利用建议框对公用特征层进行截取
        pool = self.roi(x, indices_and_rois) #公共特征层 框
        pool=QG2.forward(pool,temp) #做了一次特征调制
        fc7 = self.classifier(pool) #第四部分加一个池化
        fc7 = fc7.view(fc7.size(0), -1) #平铺
        rcnn_cls_locs = self.cls_loc(fc7) #全连接
        rcnn_scores = self.score(fc7) #全连接
        return rcnn_cls_locs, rcnn_scores #回归和分类

'''
初始化用
'''
def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()