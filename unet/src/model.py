import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False), nn.GroupNorm(8,b), nn.ReLU(inplace=True),
                         nn.Conv2d(b,b,3,padding=1,bias=False), nn.GroupNorm(8,b), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        widths=[base*2**i for i in range(5)]
        self.down=nn.ModuleList([Block(1 if i==0 else widths[i-1], w) for i,w in enumerate(widths)])
        self.up=nn.ModuleList([nn.ConvTranspose2d(widths[i+1],widths[i],2,stride=2) for i in reversed(range(4))])
        self.decode=nn.ModuleList([Block(2*widths[i],widths[i]) for i in reversed(range(4))])
        self.head=nn.Conv2d(base,3,1)
    def forward(self,x):
        skips=[]
        for i,block in enumerate(self.down):
            x=block(x)
            if i<4:
                skips.append(x); x=F.max_pool2d(x,2)
        for up,block,skip in zip(self.up,self.decode,reversed(skips)):
            x=up(x)
            if x.shape[-2:]!=skip.shape[-2:]:
                x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=block(torch.cat([skip,x],dim=1))
        return self.head(x)
