"""Inference-only final architecture; accepts RGB FP32: convolution branches [0,1], MiT [0,255].

MiT inference adaptation follows OpenMMLab MMSegmentation's Apache-2.0
MixVisionTransformer and SegformerHead (see THIRD_PARTY_NOTICES).
Parameter names intentionally match the trained checkpoint, without remapping.
Dropout and DropPath have no inference effect and are omitted from this adapter.
"""
import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

class ConvUNet(nn.Module):
    def __init__(self, family):
        super().__init__()
        encoder = 'resnet34' if family == 'resnet34' else 'tu-convnext_tiny.fb_in1k'
        self.model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self, image):
        x=(image-self.mean)/self.std
        return self.model(F.pad(x,(0,0,0,6)))[...,:378,:672]

class PatchEmbed(nn.Module):
    def __init__(self, c, d, kernel, stride):
        super().__init__()
        self.projection=nn.Conv2d(c,d,kernel,stride,padding=kernel//2)
        self.norm=nn.LayerNorm(d,eps=1e-6)
    def forward(self,x):
        x=self.projection(x);h,w=x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1,2)),(h,w)

def nchw(x,hw):return x.transpose(1,2).reshape(x.shape[0],x.shape[2],hw[0],hw[1])

class EfficientAttention(nn.Module):
    def __init__(self,d,heads,sr):
        super().__init__();self.ratio=sr
        self.attn=nn.MultiheadAttention(d,heads,dropout=0,bias=True,batch_first=False)
        if sr>1:
            self.sr=nn.Conv2d(d,d,sr,stride=sr)
            self.norm=nn.LayerNorm(d,eps=1e-6)
    def forward(self,x,hw,identity):
        kv=self.norm(self.sr(nchw(x,hw)).flatten(2).transpose(1,2)) if self.ratio>1 else x
        q=x.transpose(0,1);kv=kv.transpose(0,1)
        return identity+self.attn(q,kv,kv,need_weights=True)[0].transpose(0,1)

class MixFFN(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(d,d*4,1),nn.Conv2d(d*4,d*4,3,padding=1,groups=d*4),nn.GELU(),nn.Dropout(0),nn.Conv2d(d*4,d,1),nn.Dropout(0))
    def forward(self,x,hw,identity):
        return identity+self.layers(nchw(x,hw)).flatten(2).transpose(1,2)

class EncoderLayer(nn.Module):
    def __init__(self,d,heads,sr):
        super().__init__();self.norm1=nn.LayerNorm(d,eps=1e-6);self.attn=EfficientAttention(d,heads,sr);self.norm2=nn.LayerNorm(d,eps=1e-6);self.ffn=MixFFN(d)
    def forward(self,x,hw):
        x=self.attn(self.norm1(x),hw,x)
        return self.ffn(self.norm2(x),hw,x)

class MiTBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList();c=3
        for d,heads,depth,kernel,stride,sr in zip([64,128,320,512],[1,2,5,8],[3,4,6,3],[7,3,3,3],[4,2,2,2],[8,4,2,1]):
            self.layers.append(nn.ModuleList([PatchEmbed(c,d,kernel,stride),nn.ModuleList([EncoderLayer(d,heads,sr) for _ in range(depth)]),nn.LayerNorm(d,eps=1e-6)]));c=d
    def forward(self,x):
        outs=[]
        for patch,blocks,norm in self.layers:
            x,hw=patch(x)
            for block in blocks:x=block(x,hw)
            x=nchw(norm(x),hw);outs.append(x)
        return outs

class ConvModule(nn.Module):
    def __init__(self,c,d):
        super().__init__();self.conv=nn.Conv2d(c,d,1,bias=False);self.bn=nn.BatchNorm2d(d,eps=1e-5)
    def forward(self,x):return F.relu(self.bn(self.conv(x)),inplace=False)

class SegformerHead(nn.Module):
    def __init__(self):
        super().__init__();self.convs=nn.ModuleList([ConvModule(c,256) for c in [64,128,320,512]]);self.fusion_conv=ConvModule(1024,256);self.conv_seg=nn.Conv2d(256,2,1)
    def forward(self,inputs):
        outs=[F.interpolate(conv(x),size=inputs[0].shape[-2:],mode='bilinear',align_corners=False) for conv,x in zip(self.convs,inputs)]
        return self.conv_seg(self.fusion_conv(torch.cat(outs,dim=1)))

class MiT(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=MiTBackbone();self.decode_head=SegformerHead()
        self.register_buffer('rgb_mean',torch.tensor([123.675,116.28,103.53]).view(1,3,1,1),persistent=False)
        self.register_buffer('rgb_std',torch.tensor([58.395,57.12,57.375]).view(1,3,1,1),persistent=False)
    def forward(self,image):
        x=F.pad((image-self.rgb_mean)/self.rgb_std,(0,0,0,6))
        logits=self.decode_head(self.backbone(x))
        return F.interpolate(logits,size=(384,672),mode='bilinear',align_corners=False)[...,:378,:672]

def build(family):
    if family=='mit_b2':return MiT()
    if family not in ('resnet34','convnext_ema'):raise ValueError(family)
    return ConvUNet(family)
