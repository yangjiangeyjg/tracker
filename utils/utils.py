import numpy as np
import torch
from torch.nn import functional as F

#给定框要求计算调整变化，计算损失肯定会用到，参数为源框和目标框
#就是论文中说的先验眶和真实框
def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0] #宽，切片的时候第一维全要不切
    height = src_bbox[:, 3] - src_bbox[:, 1] #高
    ctr_x = src_bbox[:, 0] + 0.5 * width #中心横坐标
    ctr_y = src_bbox[:, 1] + 0.5 * height
    #计算目标框的长款=宽和中心坐标
    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height
    #宽和高必须是正数
    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)
    #这部分就是调整了，根据论文中的公式求得
    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)
    #把四个值按行堆叠起来再转置
    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc #框调整变化

#计算调整后的框，给定原框和调整变化参数
def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0: #没有框
        return np.zeros((0, 4), dtype=loc.dtype)
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    #计算原框的长宽和中心坐标
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height
    #切片的时候第一维全要，第二维步长设置为4
    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]
    #计算目标框的中心坐标和宽高
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    #最后计算的是左上角坐标和右下角坐标
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h
    return dst_bbox #返回调整变化后的目标框

#计算两组框的IOU
def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4: #维度不对
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2]) #左上
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:]) #右下
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

#NMS处理，一组框，有个阈值，返回一组最大的框，减少密密麻麻的框
def nms(detections_class,nms_thres=0.7):
    max_detections = []
    while np.shape(detections_class)[0]:
        # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        max_detections.append(np.expand_dims(detections_class[0],0))
        if len(detections_class) == 1:
            break
        ious = bbox_iou(max_detections[-1][:,:4], detections_class[1:,:4])[0]
        detections_class = detections_class[1:][ious < nms_thres] #大的会被删掉，小的会留下来
    if len(max_detections)==0:
        return []
    max_detections = np.concatenate(max_detections,axis=0)
    return max_detections

'''
解码，根据建议框的预测结果获得真实框
暂时先放在这里

class DecodeBox():
    def __init__(self, std, mean, num_classes):
        self.std = std #标准差
        self.mean = mean #平均值
        self.num_classes = num_classes + 1 #弹单目标跟踪这里就是2 1+1

    def forward(self, roi_cls_locs, roi_scores, rois, height, width, nms_iou, score_thresh):
        rois = torch.Tensor(rois) #rois就是建议框
        roi_cls_loc = (roi_cls_locs * self.std + self.mean) #回归
        roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])
        roi = rois.view((-1, 1, 4)).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox((roi.cpu().detach().numpy()).reshape((-1, 4)), (roi_cls_loc.cpu().detach().numpy()).reshape((-1, 4)))
        cls_bbox = torch.Tensor(cls_bbox)
        cls_bbox = cls_bbox.view([-1, (self.num_classes), 4]) #得到了框
        # 截断在宽高以内
        cls_bbox[..., 0] = (cls_bbox[..., 0]).clamp(min=0, max=width)
        cls_bbox[..., 2] = (cls_bbox[..., 2]).clamp(min=0, max=width)
        cls_bbox[..., 1] = (cls_bbox[..., 1]).clamp(min=0, max=height)
        cls_bbox[..., 3] = (cls_bbox[..., 3]).clamp(min=0, max=height)
        prob = F.softmax(torch.tensor(roi_scores), dim=1)
        raw_cls_bbox = cls_bbox.cpu().numpy()
        raw_prob = prob.cpu().numpy()
        outputs = []
        arg_prob = np.argmax(raw_prob, axis=1)
        for l in range(1, self.num_classes):
            arg_mask = (arg_prob == l)
            cls_bbox_l = raw_cls_bbox[arg_mask, l, :]
            prob_l = raw_prob[arg_mask, l]
            mask = prob_l > score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            if len(prob_l) == 0:
                continue
            label = np.ones_like(prob_l) * (l-1)
            detections_class = np.concatenate([cls_bbox_l, np.expand_dims(prob_l,axis=-1), np.expand_dims(label,axis=-1)],axis=-1)
            prob_l_index = np.argsort(prob_l)[::-1]
            detections_class = detections_class[prob_l_index]
            nms_out = nms(detections_class, nms_iou)
            if outputs==[]:
                outputs = nms_out
            else:
                outputs = np.concatenate([outputs, nms_out],axis=0)
        return outputs
'''

'''
只在训练阶段用到，训练RCNN的时候
需要选择并返回训练目标，从512个中挑选128个
正样本四分之一 负样本四分之三
'''
class ProposalTargetCreator(object):
    def __init__(self,n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi #负样本是一个区间
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape #获取真实框的数量
        roi = np.concatenate((roi, bbox), axis=0) #roi和bbox对上，行列对应
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)#预计的正样本数量
        iou = bbox_iou(roi, bbox) #计算iou
        gt_assignment = iou.argmax(axis=1)#每一个先验框对应真实框最大的iou的下标
        max_iou = iou.max(axis=1) #iou的值
        # 真实框的标签要+1因为有背景的存在
        gt_roi_label = label[gt_assignment] + 1 #加了个背景
        # 找到大于门限的真实框的索引
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)
        # 正负样本的平衡，满足建议框和真实框重合程度小于neg_iou_thresh_hi大于neg_iou_thresh_lo作为负样本
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                          neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)
        # 取出这些框对应的标签
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0 #负样本的标签都是0
        sample_roi = roi[keep_index]
        # 找到
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32)) #数据增强归一化处理
        #返回选择的roi，类别标签和调整
        return sample_roi, gt_roi_loc, gt_roi_label

'''
RPN中训练使用，选择并返回训练样本，就是框的调整和分类标签
选择的范围就是feature中密密麻麻的anchor
这里我们参考faster rcnn中的，考虑多目标，虽然是单目标跟踪，但是训练用的coco有多目标，并且论文中有多实例查询损失
计算回归损失的时候正样本参与计算，负样本不计算
'''
class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=2000,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample #样本数
        self.pos_iou_thresh = pos_iou_thresh #正阈值
        self.neg_iou_thresh = neg_iou_thresh #负阈值
        self.pos_ratio = pos_ratio #正负样本比例

    def __call__(self, bbox, anchor, img_size):
        argmax_ious, label = self._create_label(anchor, bbox) #下标和标签
        # 利用先验框和其对应的真实框进行编码求得调整
        loc = bbox2loc(anchor, bbox[argmax_ious])
        return loc, label #返回标签和调整

    def _create_label(self, anchor, bbox):
        # 1是正样本，0是负样本，-1忽略
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1) #所有标号都置为-1
        # argmax_ious为每个先验框对应的最大的真实框的序号
        # max_ious为每个真实框对应的最大的先验框的iou
        # gt_argmax_ious为每一个真实框对应的最大的先验框的序号
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox)
        # 如果小于门限函数则设置为负样本
        label[max_ious < self.neg_iou_thresh] = 0
        # 每个真实框至少对应一个先验框
        label[gt_argmax_ious] = 1
        # 如果大于门限函数则设置为正样本
        label[max_ious >= self.pos_iou_thresh] = 1
        # 判断正样本数量是否大于128，如果大于的话则去掉一些
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1 #什么也不是
        # 平衡正负样本，保持总数量为256
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        return argmax_ious, label

    def _calc_ious(self, anchor, bbox):
        # 计算所有，每个anchor和每个bbox的iou
        ious = bbox_iou(anchor, bbox)
        # 行是先验框，列是真实框，找出每一个先验框对应真实框最大的iou
        argmax_ious = ious.argmax(axis=1) #1是行，这里返回的是下标
        max_ious = ious[np.arange(len(anchor)), argmax_ious]#每一个先验框对应真实框最大的iou
        # 找到每一个真实框对应的先验框最大的iou
        gt_argmax_ious = ious.argmax(axis=0) #标号
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] # 每一个真实框对应的最大的先验框
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        #返回的是每个anchor先验眶对应真实框的最大IOU下标，每一个先验框对应真实框最大的iou ，每个真实框对应anchor的最大IOU下标
        return argmax_ious, max_ious, gt_argmax_ious
