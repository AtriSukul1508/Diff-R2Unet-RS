import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from torch import nn
import math
from diffusion import device

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                    nn.BatchNorm3d(ch_out),
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

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,time_emb_dim=32, kernal=(4,3,3), stride=(2,1,1), padding=(1,1,1), t=2,up=False):
        super(RRCNN_block,self).__init__()
        self.ch_out = ch_out
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in,ch_out,kernel_size=3,padding=1)
        self.time_mlp =  nn.Linear(time_emb_dim, ch_out)
        if up:
            self.conv1 = nn.Conv3d(2*ch_in, ch_out, 3, padding=1)
            self.transform = nn.ConvTranspose3d(ch_out, ch_out, kernal, stride, padding)
        else:
            self.conv1 = nn.Conv3d(ch_in, ch_out, 3, padding=1)
            self.transform = nn.Conv3d(ch_out, ch_out, kernal, stride, padding)
        self.conv2 = nn.Conv3d(ch_out, ch_out, 3, padding=1)
        self.bnorm1 = nn.BatchNorm3d(ch_out)
        self.bnorm2 = nn.BatchNorm3d(ch_out)
        self.relu  = nn.ReLU()
    def forward(self,x,t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 3]
        x = h + time_emb
        x = self.bnorm2(self.relu(self.conv2(x)))
        x1 = self.RCNN(x)
        return self.transform(x + x1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class R2UNet(nn.Module):
    def __init__(self,_image_channels,t=1):
        super(R2UNet,self).__init__()
        down_channels = (16,32,64,128)
        down_params = [
            [(4,5,5),(2,1,1),(1,2,2)],
            [(4,5,5),(2,1,1),(1,2,2)],
            [(4,5,5),(2,1,1),(1,2,2)],
        ]
        up_channels = (128,64,32,16)
        up_params = [
            [(4,5,5),(2,1,1),(1,2,2)],
            [(4,5,5),(2,1,1),(1,2,2)],
            [(4,5,5),(2,1,1),(1,2,2)],
        ]
        time_emb_dim = 32
        self.downs = nn.ModuleList([RRCNN_block(ch_in=down_channels[i], ch_out=down_channels[i+1], time_emb_dim=time_emb_dim, \
                                     kernal=down_params[i][0], stride=down_params[i][1], padding=down_params[i][2],t=t) \
                    for i in range(len(down_channels)-1)])
        self.RCNNups = nn.ModuleList([RRCNN_block(up_channels[i], up_channels[i+1], time_emb_dim, \
                                     up_params[i][0], up_params[i][1], up_params[i][2],t=t,up=True) \
                    for i in range(len(up_channels)-1)])


        self.Conv_1x1 = nn.Conv3d(up_channels[-1],_image_channels,kernel_size=1,stride=1,padding=0)
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        self.hetconv_layer = nn.Sequential(
            HetConv2D(_image_channels, 16,
                p = 1,
                g = 1,
                ),
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
        return self.Conv_1x1(x)
    def return_features(self):
        temp_features = []
        temp_features = self.features[:]
        self.features = []
        return temp_features



if __name__ == "__main__":
    t = torch.full((1,), 100,  dtype=torch.long).to(device)
    x = torch.randn((100,1,104,16,16)).to(device)
    model = R2UNet(_image_channels=1).to(device)
    out = model(x,t)
    print('output',out.shape)