import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.transformer.transformer_predictor import MLP

    
class Double_ConvBnRule(nn.Module):

    def __init__(self, in_channels, out_channels=64):
        super(Double_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.resconv=nn.Sequential(nn.Conv2d(in_channels,out_channels,1),nn.BatchNorm2d(out_channels))

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu1(feat)
        feat = self.conv2(feat)
        feat = self.bn2(feat)+self.resconv(x)
        feat = self.relu2(feat)

        return feat
    
class Double_ConvBnRule_CBAM(nn.Module):

    def __init__(self, in_channels, out_channels=64):
        super(Double_ConvBnRule_CBAM, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.resconv=nn.Sequential(nn.Conv2d(in_channels,out_channels,1),nn.BatchNorm2d(out_channels))
        self.cbam=CBAM(out_channels)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu1(feat)
        feat = self.conv2(feat)
        feat = self.bn2(feat)
        feat = self.cbam(feat)+self.resconv(x)
        feat = self.relu2(feat)
        return feat    
    

def fHb(freq=4,lf_mask=0,hf_mask=0):
    
    freq=np.uint8(freq)
    if freq>8:freq=8
    f = torch.zeros(1, 64, 1, 1)+torch.tensor([1-hf_mask])
    for i in range(freq):
        for j in range(freq):
            f[0, i+j*8, 0, 0]=lf_mask
    f[0, 0, 0, 0]=0
    
    return torch.bernoulli(f.repeat(1,3,1,1))

    

def fLb(freq=4,lf_mask=0,hf_mask=0):

    freq=np.uint8(freq)
    if freq>8:freq=8
    f = torch.zeros(1, 64, 1, 1)+torch.tensor([hf_mask])
    for i in range(freq):
        for j in range(freq):
            f[0, i+j*8, 0, 0]=1-lf_mask
    f[0, 0, 0, 0]=1

    return torch.bernoulli(f.repeat(1,3,1,1))


class HGAM(nn.Module):

    def __init__(self, dim_spa,dim_freq):
        super(HGAM, self).__init__()
        
        self.conv=nn.Sequential(nn.Conv2d(dim_spa+dim_freq,dim_spa,3,padding=1),nn.BatchNorm2d(dim_spa),nn.ReLU(inplace=True))
        self.freq_conv=Double_ConvBnRule_CBAM(192,64)
 
    def forward(self, spa, freq):

        freq=self.freq_conv(freq)
        freq = torch.nn.functional.interpolate(freq,spa.size()[2:],mode='bilinear',align_corners=True)
        out=self.conv(torch.cat([spa,freq],1))
        return out
    
class DRSM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channel = in_channels
        self.out_channel = out_channels
        self.conv_mask = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1, bias=True)
        self.sm = nn.Softmax2d()
        self.conv_out=nn.Conv2d(out_channels*4,out_channels,3,padding=1)
        self.bn_out=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x, kernel):

        b, c, h, w = x.size()
        masks = self.conv_mask(x)
        masks=self.sm(masks)

        masks0=masks
        x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)  
        x = x * masks.unsqueeze(2)
        x = x.view(1, -1, h, w)  
        
        k1=kernel[:,0:3,:,:,:,:]
        k3=kernel[:,0:3,:,:,:,:]
        k5=kernel[:,0:3,:,:,:,:]
        k7=kernel[:,0:3,:,:,:,:]
        
        k1=k1.reshape(-1, self.in_channel, 3, 3)
        k3=k3.reshape(-1, self.in_channel, 3, 3)
        k5=k5.reshape(-1, self.in_channel, 3, 3)
        k7=k7.reshape(-1, self.in_channel, 3, 3)

        x1 = F.conv2d(x, k1, stride=1, padding=1, groups=b * 3).view(b, 3, self.out_channel, h, w)
        x1= torch.sum(x1 , dim=1)

        x3 = F.conv2d(x, k3, stride=1, padding=3, groups=b * 3,dilation=3).view(b, 3, self.out_channel, h, w)
        x3= torch.sum(x3 , dim=1)     

        x5 = F.conv2d(x, k5, stride=1, padding=5, groups=b * 3,dilation=5).view(b, 3, self.out_channel, h, w)
        x5= torch.sum(x5 , dim=1)  

        x7 = F.conv2d(x, k7, stride=1, padding=7, groups=b * 3,dilation=7).view(b, 3, self.out_channel, h, w)
        x7= torch.sum(x7 , dim=1)

        out=self.conv_out(torch.cat([x1,x3,x5,x7],1))
        out=self.bn_out(out)
        out=self.relu(out)
        
        return out, masks0
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    
class QueryGeneration(nn.Module):
    def __init__(self,query_num,embedding_dim) -> None:
        super().__init__()
        self.emb = nn.Embedding(44**2, embedding_dim)
        
        self.conv_fl=Double_ConvBnRule(64,query_num*4)
        self.mlp=MLP(embedding_dim,embedding_dim*2,embedding_dim,2)
        self.query_num=query_num

    def forward(self,feat_l):

        feat_l=self.conv_fl(feat_l)
        feat_l=feat_l.flatten(2)
        query=torch.einsum('bnm, mc -> bnc',feat_l,self.emb.weight)
        query=self.mlp(query)

        return query