
import torch.nn as nn

'''
x是待检测的帧提取的特征图     h*w的
z是第一帧进行框截取后提取的特征图又进行了ROIAlign k*k的
'''

class QG1(nn.Module):
    # k是7 c就是256
    def __init__(self,k=7,c=256):
        super(QG1, self).__init__()
        self.fx = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1,padding=1),
        )
        self.fz = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=k, stride=1, padding=0),
        )
        self.fout = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0),
        )
        self.conv=nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
    def forward(self,x,z):
        x=self.fx(x)
        z=self.fz(z)
        self.conv.weight = nn.Parameter(z)
        out=self.conv(x)
        out=self.fout(out)
        return out

class QG2(nn.Module):
    def __init__(self,c=256):
        super(QG1, self).__init__()
        self.hx = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1,padding=1),
        )
        self.hz = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
        )
        self.hout = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0),
        )
        self.conv=nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
    def forward(self,x,z):
        x=self.hx(x)
        z=self.hz(z)
        self.conv.weight = nn.Parameter(z)
        out=self.conv(x)
        out=self.hout(out)
        return out