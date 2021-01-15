import os
import random
from os import path

'''
为数据制作索引，每一个文件夹中采取总帧数/20的帧对，帧对
是随机采取的。每行是图片路径和bbox，每两行作为一组帧对
'''
if __name__=="__main__":
    file = open("H:\\LaSOTTest\\train.txt", 'a')
    files=os.listdir("H:\\LaSOTTest\\zip")
    for f in files:
        real_path=path.join("H:\\LaSOTTest\\zip",f,"img")
        filess=os.listdir(real_path)
        num=len(filess)
        numm=int(num/20)
        for i in range(numm):
            frame1= random.randint(1, num)
            framee1=str(frame1).zfill(8)
            frame2= random.randint(1, num)
            framee2 = str(frame2).zfill(8)
            path1=path.join("H:\\LaSOTTest\\zip",f,"img",framee1+".jpg")
            path2=path.join("H:\\LaSOTTest\\zip",f,"img",framee2+".jpg")
            txt=path.join("H:\\LaSOTTest\\zip",f,"groundtruth.txt")
            with open(txt, "r", encoding='utf-8') as ff:
                bbox = ff.readlines()
                file.write(path1+' '+bbox[frame1-1]+path2+' '+bbox[frame2-1])
    file.close()
