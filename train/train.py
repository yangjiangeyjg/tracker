
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

from train.trainer import globaltracker_trainer
from nets.globaltrack import globaltrack
from utils.dataloader import globaltrackDataset, globaltrack_dataset_collate

'''
学习率0.01 第8代和第11代衰减0.1
动量0.9 权值衰减10 -4
0.4 0.4 0.2 COCO GOT-10K LaSOT
12+12代，COCO预训练的faster rcnn初始化参数
三个数据集训练，四个基准数据集测试 对优化器进行设置就好了
'''


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_ont_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    rcnn_loc_loss = 0
    rcnn_cls_loss = 0
    val_toal_loss = 0
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            temp,imgs, boxes, labels = batch[0], batch[1], batch[2],batch[3]
            with torch.no_grad():
                if cuda:
                    temp=Variable(torch.from_numpy(temp).type(torch.FloatTensor)).cuda()
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                    boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
                    labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]
                else:
                    temp = Variable(torch.from_numpy(temp).type(torch.FloatTensor))
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                    boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)) for box in boxes]
                    labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)) for label in labels]
            losses = train_util.train_step(temp,imgs, boxes, labels, 1)
            rpn_loc, rpn_cls, rcnn_loc, rcnn_cls, total = losses
            total_loss += total
            rpn_loc_loss += rpn_loc
            rpn_cls_loss += rpn_cls
            rcnn_loc_loss += rcnn_loc
            rcnn_cls_loss += rcnn_cls
            pbar.set_postfix(**{'total': total_loss.item() / (iteration + 1),
                                'rpn_loc': rpn_loc_loss.item() / (iteration + 1),
                                'rpn_cls': rpn_cls_loss.item() / (iteration + 1),
                                'roi_loc': rcnn_loc_loss.item() / (iteration + 1),
                                'roi_cls': rcnn_cls_loss.item() / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            temp,imgs, boxes, labels = batch[0], batch[1], batch[2],batch[3]
            with torch.no_grad():
                if cuda:
                    temp = Variable(torch.from_numpy(temp).type(torch.FloatTensor)).cuda()
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                    boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
                    labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]
                else:
                    temp = Variable(torch.from_numpy(temp).type(torch.FloatTensor))
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                    boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)) for box in boxes]
                    labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)) for label in labels]

                train_util.optimizer.zero_grad()
                losses = train_util.forward(temp,imgs, boxes, labels, 1)
                _, _, _, _, val_total = losses
                val_toal_loss += val_total
            pbar.set_postfix(**{'total_loss': val_toal_loss.item() / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'E:PY//pypy//A_my_tracker//logs//Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))






if __name__ == "__main__":
    annotation_path="H://LaSOTTest//train.txt"
    NUM_CLASSES = 1
    IMAGE_SHAPE = [600, 600, 3]
    BACKBONE = "resnet50"
    model = globaltrack(NUM_CLASSES, backbone=BACKBONE)

    model_path = ''
    print('Loading weights into state dict...')
    Cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict() #模型的字典对象，每一层和对应的参数的映射
    pretrained_dict = torch.load(model_path, map_location=device) #模型的预训练参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict) #更新
    model.load_state_dict(model_dict)#模型加载预训练的参数
    print('Finished!')
    net = model.train()
    '''
        if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    '''
    val_split=0.1
    net = net.cuda()
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    lr = 1e-2
    Init_Epoch = 0
    Freeze_Epoch = 5
    optimizer = optim.Adam(net.parameters(), lr, weight_decay=10e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    train_dataset = globaltrackDataset(lines[:num_train], (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    val_dataset = globaltrackDataset(lines[num_train:], (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    gen = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=0, pin_memory=True,
                     drop_last=True, collate_fn=globaltrack_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=globaltrack_dataset_collate)
    epoch_size = num_train
    epoch_size_val = num_val

    for param in model.extractor.parameters():
        param.requires_grad = False
    model.freeze_bn()
    train_util = globaltracker_trainer(model, optimizer)
    for epoch in range(Init_Epoch, Freeze_Epoch):
        fit_ont_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
        lr_scheduler.step()

    lr = 1e-3
    Freeze_Epoch = 5
    Unfreeze_Epoch = 10
    optimizer = optim.Adam(net.parameters(), lr, weight_decay=10e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    train_dataset = globaltrackDataset(lines[:num_train], (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    val_dataset = globaltrackDataset(lines[num_train:], (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        gen = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=globaltrack_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=globaltrack_dataset_collate)
    epoch_size = num_train
    epoch_size_val = num_val
    for param in model.extractor.parameters():
        param.requires_grad = True
    model.freeze_bn()
    train_util = globaltracker_trainer(model, optimizer)
    for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
        fit_ont_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
        lr_scheduler.step()
