
from PIL import Image
import cv2
'''
截取图片，我们以LaSOT数据集为例试一下
其它数据集我们就根据格式处理解码吧，处理解码后获得bbox后再调用该函数
bbox的四个值是 左上顶点横 左上顶点纵 宽 高
'''
def template_clip(img,bbox):
    #img = Image.open("path")
    cropped = img.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
    return cropped
'''
测试函数的正确性
if __name__ == "__main__":
    img = Image.open("H://LaSOTTest//zip//dog-7//img//00000010.jpg")
    print(img.size)
    cropped = img.crop((464, 109, 723, 255))  # (left, upper, right, lower)
    #cv2.imwrite("C://Users//lenovo\Desktop//55.jpg", cropped)
    cropped.save("C://Users//lenovo//Desktop//55.jpg")
'''