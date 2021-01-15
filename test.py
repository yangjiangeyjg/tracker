
import torch
import torch.nn as nn
from PIL import Image
from A_my_tracker.netss.globaltrack import globaltrack
from A_my_tracker.netss.template_clip import template_clip

'''
简单测试一下各个模型
'''
num_classes=1 #COCO数据集不是，其它都为1,看看怎么加载参数吧
model=globaltrack(num_classes,"predict")
model_path1='E://py//pypy//A_my_tracker//model//qg_rcnn_r50_fpn_2x_20181010-443129e1.pth' #作者采取的预训练的faster rcnn模型
model_path2='E://py//pypy//A_my_tracker//model//qg_rcnn_r50_fpn_coco_got10k_lasot.pth'  #作者训练好的模型，globaltrack的
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(model_path2, map_location=device)
#model.load_state_dict(state_dict) #暴力加载参数
#print(model_path2)

'''
imgk = input('Input template filename:')
imgz = Image.open(imgk)
#image = Image.open(imgz)
boxx=[18,39,68,77]
#这里假设框是四元组列表，这部分应该是不同的数据有着不同的处理选择
imgz=template_clip(imgz,boxx) #将模板帧也就是第一帧根据框给截取出来

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = globaltrack.detect_image(image,imgz) #打印图片显示跟踪的框，可视化跟踪结果，待写
        r_image.show()
'''