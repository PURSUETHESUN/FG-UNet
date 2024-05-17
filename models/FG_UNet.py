from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import pvt_v2
from timm.models.vision_transformer import _cfg

#dilation convlution
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,groups=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=True), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sharedMLP(self.avg_pool(x) + self.max_pool(x))
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

class EAM(nn.Module):
    def __init__(self,in_channel):
        super(EAM, self).__init__()
        self.sobel_x1, self.sobel_y1 = get_sobel(64, 1)# 64 * 4
        self.sobel_x2, self.sobel_y2 = get_sobel(64, 1)# 128 * 4
        self.sobel_x3, self.sobel_y3 = get_sobel(64, 1)# 192 * 4
        self.sobel_x4, self.sobel_y4 = get_sobel(64, 1)# 256 * 4

        self.block = nn.Sequential(
            BasicConv2d(in_planes=in_channel, out_planes=1, kernel_size=3,stride=1,padding=1),
        )

    def forward(self, x4, x3, x2, x1):

        size = x1.size()[2:]
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)

        s1 = run_sobel(self.sobel_x1, self.sobel_y1, x1)
        s2 = run_sobel(self.sobel_x2, self.sobel_y2, x2)
        s3 = run_sobel(self.sobel_x3, self.sobel_y3, x3)
        s4 = run_sobel(self.sobel_x4, self.sobel_y4, x4)

        out = self.block(torch.cat((s4, s3, s2, s1), dim=1))

        return out

class Dual_Attention(nn.Module):
    def __init__(self,
                 embed_dims,
                 kernel_size=3):
        super(Dual_Attention, self).__init__()
        self.embed_dims = embed_dims
        mid_dim = 4 * embed_dims
        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,stride=1,padding=0,bias=True)
        self.act = nn.GELU()
        self.ca = ChannelAttention(embed_dims)
        self.sp = SpatialAttention()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dims,out_channels=embed_dims,kernel_size=3,padding=1,groups=embed_dims),
            Conv1x1(inplanes=embed_dims,planes=embed_dims),
            Conv1x1(inplanes=embed_dims,planes=embed_dims)
        )
        self.fc2 = nn.Conv2d(
            in_channels= embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x1 = self.ca(x) * x
        x2 = self.sp(x) * x
        x3 = self.dwconv(x1 * x2)
        x = self.fc2(x1 + x2 + x3 )

        return x

class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = x
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.DW_conv0(x)
        x = self.PW_conv(x)
        return x

class MultiOrderGatedAggregation(nn.Module):
    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.group_num = embed_dims // 4
        # self.proj_1 = nn.Conv2d(
        #     in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1,padding=0)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=embed_dims//4, out_channels=embed_dims, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj_2 = Conv1x1(inplanes=embed_dims, planes=embed_dims)

        # activation for gating and valu
        self.act_gate = nn.Sigmoid()

    def forward(self, x):

        g = self.gate(x)
        v = self.value(x)
        xg = self.act_gate(g) * v
        xl = self.g(x.mean(dim=(2,3),keepdim=True))
        x = self.proj_2(xg+xl)
        return x

class MSCA(nn.Module):
    def __init__(self, inchannel):
        super(MSCA, self).__init__()
        self.MOGA = MultiOrderGatedAggregation(embed_dims = inchannel)
        self.DA = Dual_Attention(embed_dims = inchannel)

    def forward(self, x):
        x = self.MOGA(x) + x
        x = self.DA(x) + x
        return x

class Dual_Guide(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Dual_Guide, self).__init__()
        mid_dim = in_channel//2
        self.obj = nn.Sequential(
            Conv1x1(inplanes=3*mid_dim, planes=mid_dim),
            BasicConv2d(in_planes=mid_dim, out_planes=mid_dim,kernel_size=3,stride=1,padding=1,groups=mid_dim),
            Conv1x1(inplanes=mid_dim, planes = mid_dim),
            Conv1x1(inplanes=mid_dim, planes=mid_dim)
        )
        self.bck = nn.Sequential(
            Conv1x1(inplanes=3*mid_dim, planes=mid_dim),
            BasicConv2d(in_planes=mid_dim, out_planes=mid_dim, kernel_size=3, stride=1, padding=1,groups=mid_dim),
            Conv1x1(inplanes=mid_dim, planes= mid_dim),
            Conv1x1(inplanes= mid_dim, planes=mid_dim)
        )
        self.bou = nn.Sequential(
            Conv1x1(inplanes = 2*mid_dim, planes = mid_dim),
            BasicConv2d(in_planes=mid_dim, out_planes=mid_dim, kernel_size=3, stride=1, padding=1,groups=mid_dim),
            Conv1x1(inplanes=mid_dim, planes= mid_dim),
            Conv1x1(inplanes= mid_dim, planes=mid_dim)
        )

        self.dual_att = Dual_Attention(embed_dims=3*mid_dim)
        self.re = Conv1x1(inplanes=3*mid_dim, planes=64)

    def forward(self, x_edg, x_obj):

        x_obj = torch.sigmoid(x_obj)

        x_bound = 1 - torch.abs_(x_obj-0.5)/0.5
        #branch1
        x_o = self.obj(torch.cat([x_edg ,x_bound, x_obj], dim=1)) + x_obj

        #branch2
        x_back = -1 * x_obj + 1
        x_b = self.bck(torch.cat([x_edg, x_bound, x_back], dim=1)) + x_back

        #
        x = self.bou(torch.cat([x_o, x_b], dim=1))
        x_bound = 1 - torch.abs_(torch.sigmoid(x) - 0.5) / 0.5

        x = self.dual_att(torch.cat([x_o, x_bound, x_b], dim=1))
        x = self.re(x)
        return x

class backbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = pvt_v2.pvt_v2_b2()

        checkpoint = torch.load("./pre_weight//pvt_v2_b2.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.MS = nn.ModuleList([])
        self.MS.append(BasicConv2d(in_planes=64+128,out_planes=64,kernel_size=3, stride=1,padding=1))
        self.MS.append(BasicConv2d(in_planes=64+128+320,out_planes=64,kernel_size=3,stride=1,padding=1))
        self.MS.append(BasicConv2d(in_planes=64+320+512, out_planes=64, kernel_size=3, stride=1, padding=1))
        self.MS.append(BasicConv2d(in_planes=64+512, out_planes=64, kernel_size=3, stride=1, padding=1))

        self.g1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

        self.g21 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.g22 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

        self.g31 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.g32 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

        self.g4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.SFA = nn.ModuleList([])
        for i in range(4):
            self.SFA.append(nn.Sequential(MSCA(64)))

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:#patch_embeding层
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        p1, p2, p3, p4 = self.get_pyramid(x)
        p21 = F.interpolate(input=p2, size=p1.size()[2:], mode="bilinear", align_corners=False)
        x1 = self.MS[0](torch.cat([p1, p21], dim=1))
        x1 = self.SFA[0](x1)
        x1 = x1 * self.g1(p21)

        p12 = F.interpolate(input=x1,size=p2.size()[2:],mode="bilinear",align_corners=False)
        p32 = F.interpolate(input=p3,size=p2.size()[2:],mode="bilinear",align_corners=False)
        x2 = self.MS[1](torch.cat([p12, p2, p32], dim=1))
        x2 = x2 * self.g21(p12)
        x2 = self.SFA[1](x2)
        x2 = x2 * self.g22(p32)

        p23 = F.interpolate(input=x2,size=p3.size()[2:],mode="bilinear", align_corners=False)
        p43 = F.interpolate(input=p4,size=p3.size()[2:],mode="bilinear", align_corners=False)
        x3 = self.MS[2](torch.cat([p23, p3, p43], dim=1))
        x3 = x3 * self.g31(p23)
        x3 = self.SFA[2](x3)
        x3 = x3 * self.g32(p43)

        p34 = F.interpolate(input=x3, size=p4.size()[2:], mode="bilinear", align_corners=False)
        x4 = self.MS[3](torch.cat([p34, p4], dim=1))
        x4 = x4 * self.g4(p34)
        x4 = self.SFA[3](x4)

        return x1, x2, x3, x4

class FG(nn.Module):
    def __init__(self,seg_classes=1):
        super(FG, self).__init__()
        self.backbone = backbone()

        self.num_class = seg_classes
        self.eam = EAM(in_channel=256)

        self.obj4 = nn.Conv2d(64,1,kernel_size=1)
        self.obj3 = nn.Conv2d(64,1,kernel_size=1)
        self.obj2 = nn.Conv2d(64,1,kernel_size=1)
        self.obj1 = nn.Conv2d(64,1,kernel_size=1)

        self.dg1 = Dual_Guide(in_channel=64+64, out_channel=64)
        self.dg2 = Dual_Guide(in_channel=64+64, out_channel=64)
        self.dg3 = Dual_Guide(in_channel=64+64, out_channel=64)
        self.dg4 = Dual_Guide(in_channel=64+64, out_channel=64)

        self.predictor_obj4 = BasicConv2d(in_planes=64,out_planes=16,kernel_size=1,padding=0)
        self.predictor_obj3 = BasicConv2d(in_planes=64+16,out_planes=32,kernel_size=1,padding=0)
        self.predictor_obj2 = BasicConv2d(in_planes=64+32,out_planes=48,kernel_size=1,padding=0)
        self.predictor1 = nn.Conv2d(64+6, self.num_class, 1)
        self.predictor2 = nn.Conv2d(64, self.num_class, 1)
        self.predictor3 = nn.Conv2d(64, self.num_class, 1)
        self.predictor4 = nn.Conv2d(64, self.num_class, 1)


    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)  # [B, 64, 88, 88]  [B, 64, 44, 44]   [B, 64, 22, 22]   [B, 64, 11, 11]

        edge = self.eam(x4, x3, x2, x1)
        edge_att = torch.sigmoid(edge) # [B, 1, 88, 88]

        x1e = x1 * edge_att
        edge_att2 = F.interpolate(edge_att, x2.size()[2:], mode='bilinear', align_corners=False)
        x2e = x2 * edge_att2
        edge_att3 = F.interpolate(edge_att, x3.size()[2:], mode='bilinear', align_corners=False)
        x3e = x3 * edge_att3
        edge_att4 = F.interpolate(edge_att, x4.size()[2:], mode='bilinear', align_corners=False)
        x4e = x4 * edge_att4

        # dual_guid_attention
        c4_att = self.obj4(x4) # [B,1,11,11]
        x4o = x4 * torch.sigmoid(c4_att) # [B, 64, 11, 11]
        c4 = self.dg4(x4e, x4o) # [16, 64, 11, 11]

        c4 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False) #[16, 64, 11, 11]
        c3_att = self.obj3(c4)# [16,1,22,22]
        x3o = c4 * torch.sigmoid(c3_att)   # [16,192,22,22]
        c3 = self.dg3(x3e, x3o)  # [16,64,22,22]

        c3 = F.interpolate(c3, scale_factor=2, mode='bilinear', align_corners=False)  # [16, 64, 44, 44]
        c2_att = self.obj2(c3) # [16,1,44,44]
        x2o = c3 * torch.sigmoid(c2_att) # [16,128,44,44]
        c2 = self.dg2(x2e, x2o)  # [16,64,44,44]

        c2 = F.interpolate(c2, scale_factor=2, mode='bilinear', align_corners=False)  # [16, 64, 88, 88]
        c1_att = self.obj1(c2) # [16,1,88,88]
        x1o = c2 * torch.sigmoid(c1_att)  # [16,64,88,88]
        c1 = self.dg1(x1e, x1o)  # [16,64,88,88]

        c1 = F.interpolate(c1, scale_factor=4, mode='bilinear', align_corners=False)  # [16,64,88,88] -> [16,64,352,352]
        ob4 = F.interpolate(c4, scale_factor=16, mode='bilinear', align_corners=False)  # [16,64,352,352]
        ob3 = F.interpolate(c3, scale_factor=8, mode='bilinear', align_corners=False)  # [16,64,352,352]
        ob2 = F.interpolate(c2, scale_factor=4, mode='bilinear', align_corners=False)  # [16,64,352,352]
        output2 = self.predictor2(ob2)
        output3 = self.predictor3(ob3)
        output4 = self.predictor4(ob4)

        output1 = self.predictor1(torch.cat([c1,output2,output3,output4],dim=1))

        return output1, output2, output3, output4

if __name__ == '__main__':
    img_size = 352

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 3, img_size, img_size).to(device)

    model = FG(seg_classes=2).to(device)
    print(model);
    out1, out2, out3, out4 = model(x)
    print(out1.size())
    # print(out.size())
    print(type(out1))
    # -- coding: utf-8 --
    # import torch
    import torchvision
    from thop import profile
    Flops, params = profile(model, inputs=(x,))  # macs
    print('Flops: % .4fG' % (Flops / 1000000000))
    print('params: % .4fM' % (params / 1000000))
