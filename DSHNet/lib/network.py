import torch
import torchvision.models as models
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from lib.transformer.transformer_predictor import TransformerPredictor
from lib.blocks import *
from lib.pvtv2 import pvt_v2_b2


import numpy as np

class DSHNet(nn.Module):
    def __init__( self):
        super(DSHNet, self).__init__()

        self.backbone = pvt_v2_b2()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.channel_layer1 = Double_ConvBnRule(64)
        self.channel_layer2 = Double_ConvBnRule(128)
        self.channel_layer3 = Double_ConvBnRule(320)
        self.channel_layer4 = Double_ConvBnRule(512)

        self.drsm4 = DRSM(64, 64)
        self.drsm3 = DRSM(64, 64)
        self.drsm2 = DRSM(64, 64)
        self.drsm1 = DRSM(64, 64)
        

        self.skip_connect_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
 
        self.skip_connect_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip_connect_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        
        self.predict_layer_4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True)
        self.predict_layer_3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True)
        self.predict_layer_2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True)
        self.predict_layer_final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True)

        self.querys_gen =QueryGeneration(3,128)
        self.shkg = TransformerPredictor(in_channels=512,hidden_dim=128,num_queries=3,nheads=8,dropout=0.1,dim_feedforward=2048,enc_layers=0,
                                         dec_layers=6,mask_dim=256,pre_norm=False,deep_supervision=True,enforce_input_project=True,base_c=64,num_stage=4)


        self.HGAM1 = HGAM(64,64)
        self.HGAM2 = HGAM(64,64)
        self.HGAM3 = HGAM(64,64)
        self.HGAM4 = HGAM(64,64)

        self.conv_l=nn.Conv2d(192,64,1)

        if self.training:
            self.initialize_weights()

    def forward(self, x,x_DCT=None):


        x1,x2,x3,x4 = self.backbone(x)

        x1_t = self.channel_layer1(x1)
        x2_t = self.channel_layer2(x2)
        x3_t = self.channel_layer3(x3)
        x4_t = self.channel_layer4(x4)

        if self.training: 
            fhb1=fHb(3,0.1,0.1).to(x_DCT.device)
            fhb2=fHb(3,0.1,0.1).to(x_DCT.device)
            fhb3=fHb(3,0.1,0.1).to(x_DCT.device)
            fhb4=fHb(3,0.1,0.1).to(x_DCT.device)
            flb=fLb(3,0.1,0.1).to(x_DCT.device)

        else:
            fhb1=fHb(3,0,0).to(x_DCT.device)
            fhb2=fHb(3,0,0).to(x_DCT.device)  
            fhb3=fHb(3,0,0).to(x_DCT.device)
            fhb4=fHb(3,0,0).to(x_DCT.device)
            flb=fLb(3,0,0).to(x_DCT.device)


        freq_l=x_DCT*(flb)
        freq_l=self.conv_l(freq_l)
        query = self.querys_gen(freq_l)
        rhk4, rhk3, rhk2, rhk1 = self.shkg(x4, query)
        
        freq_h1=x_DCT*(fhb1)
        freq_h2=x_DCT*(fhb2)
        freq_h3=x_DCT*(fhb3)
        freq_h4=x_DCT*(fhb4)
        
        x1_t = self.HGAM1(x1_t, freq_h1)
        x2_t = self.HGAM2(x2_t, freq_h2)
        x3_t = self.HGAM3(x3_t, freq_h3)
        x4_t = self.HGAM4(x4_t, freq_h4)
        
        x4_t = self.upsample2(x4_t)
        x4_u, x4_mask = self.drsm4(x4_t, rhk4)
        mid_pred_4 = self.predict_layer_4(x4_u)

        c4_u = torch.cat((x4_u, x3_t), dim=1)
        x4_u = self.skip_connect_conv4(c4_u)

        x4_u = self.upsample2(x4_u)
        x3_u, x3_mask = self.drsm3(x4_u, rhk3)
        mid_pred_3 = self.predict_layer_3(x3_u)

        c3_u = torch.cat((x3_u, x2_t), dim=1)
        x3_u = self.skip_connect_conv3(c3_u)

        x3_u = self.upsample2(x3_u)
        x2_u,x2_mask = self.drsm2(x3_u, rhk2)
        mid_pred_2 = self.predict_layer_2(x2_u)

        c2_u = torch.cat((x2_u, x1_t), 1)
        x2_u = self.skip_connect_conv2(c2_u)
        
        x2_u = self.upsample2(x2_u)
        x1_u, x1_mask = self.drsm1(x2_u, rhk1)
        x1_u=self.upsample2(x1_u)

        
        pred = self.predict_layer_final(x1_u)

        masks = [x4_mask, x3_mask, x2_mask, x1_mask]
        mid_preds = [mid_pred_4,mid_pred_3, mid_pred_2]

        return pred,masks,mid_preds



    def initialize_weights(self):
        path = './pretrain/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

