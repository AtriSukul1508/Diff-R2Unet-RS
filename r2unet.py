import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from torch import nn
import math
from diffusion import device

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class RecurrentBlock(nn.Module):
    def __init__(self,out_channel,t=2):
        super(RecurrentBlock,self).__init__()
        self.t = t
        self.out_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv3d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=True),
                    nn.BatchNorm3d(out_channel),
                        nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)

            x1 = self.conv(x+x1)
        return x1
class HetConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = None, bias = None,p = 64, g = 64):
        super(HetConv2D, self).__init__()
        # Groupwise Convolution
        self.groupwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,groups=g,padding = kernel_size//3, stride = stride)
        # Pointwise Convolution
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,groups=p, stride = stride)
    def forward(self, x):
        return self.groupwise_conv(x) + self.pointwise_conv(x)

class RRConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,time_emb_dim=32, kernal=(4,3,3), stride=(2,1,1), padding=(1,1,1), t=2,upsample=False):
        super(RRConvBlock,self).__init__()
        self.RCNN = RecurrentBlock(out_channel,t=t)
        self.time_mlp =  nn.Linear(time_emb_dim, out_channel)
        if upsample:
            self.conv1 = nn.Conv3d(2*in_channel, out_channel, 3, padding=1)
            self.transform = nn.ConvTranspose3d(out_channel, out_channel, kernal, stride, padding)
        else:
            self.conv1 = nn.Conv3d(in_channel, out_channel, 3, padding=1)
            self.transform = nn.Conv3d(out_channel, out_channel, kernal, stride, padding)
        self.conv2 = nn.Conv3d(out_channel, out_channel, 3, padding=1)
        self.bnorm1 = nn.BatchNorm3d(out_channel)
        self.bnorm2 = nn.BatchNorm3d(out_channel)
        self.relu  = nn.ReLU()
    def forward(self,x,t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 3]
        x = h + time_emb
        x = self.bnorm2(self.relu(self.conv2(x)))
        x1 = self.RCNN(self.RCNN((x)))
        return self.transform(x + x1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeds = math.log(10000) / (half_dim - 1)
        embeds = torch.exp(torch.arange(half_dim, device=device) * -embeds)
        embeds = time[:, None] * embeds[None, :]
        return torch.cat((embeds.sin(), embeds.cos()), dim=-1)


class R2UNet(nn.Module):
    def __init__(self,_image_channels,t=1):
        super(R2UNet,self).__init__()
        channels = [16,32,64,128]
        params = [(4,5,5),(2,1,1),(1,2,2)]
        time_emb_dim = 32
        self.downs = nn.ModuleList([RRConvBlock(in_channel=channels[i], out_channel=channels[i+1], time_emb_dim=time_emb_dim,kernal=params[0], stride=params[1], padding=params[2],t=t) \
                    for i in range(0,len(channels)-1)])
        self.RCNNups = nn.ModuleList([RRConvBlock(channels[3-i], channels[3-(i+1)], time_emb_dim,params[0], params[1], params[2],t=t,upsample=True) \
                    for i in range(0,len(channels)-1)])


        self.out_conv = nn.Conv3d(channels[0],_image_channels,kernel_size=1,stride=1,padding=0)
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        self.hetconv_layer = nn.Sequential(
            HetConv2D(_image_channels, 16,p = 1,g = 1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        self.features = []

    def forward(self,x,timestep,feature=False):
        t = self.time_mlp(timestep)
        bands = x.shape[2]
        x = x.reshape(x.shape[0],1,bands,16*16)
        x = self.hetconv_layer(x)
        x = x.reshape(x.shape[0],-1,bands,16,16)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for rcnn_up in self.RCNNups:
            residual_x = residual_inputs.pop()
            x = torch.cat((residual_x,x),dim=1)
            if feature:
                self.features.append(x.detach().cpu().numpy())
            x = rcnn_up(x,t)
        return self.out_conv(x)



if __name__ == "__main__":
    t = torch.full((1,), 100,  dtype=torch.long).to(device)
    x = torch.randn((100,1,104,16,16)).to(device)
    model = R2UNet(_image_channels=1).to(device)
    out = model(x,t)
    print('output',out.shape)
