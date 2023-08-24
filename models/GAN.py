import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import numpy.random as random


class _NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
        super(_NetG, self).__init__()
        self.ngf = ngf# 生成器feature map数(计算通道数量的一倍增量值)
        # input noise (batch_size, 100)
        # fc层输出的结果进入第一个size为4的df层（（ngf*8）*（4*4））------ngf*2**n为不同df层的通道扩充
        self.fc = nn.Linear(nz, ngf*8*4*4)
        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        # [i，o]：[8,8][8,8][8,8][8,4][4,2][2,1]
        # channel_nums：[nf * i，nf * o]
        in_out_pairs = get_G_in_out_chs(ngf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.GBlocks.append(G_Block(cond_dim, in_ch, out_ch, upsample=True))
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            #最后一个是32 * 1的out_ch，通过卷积提取特征生成3个通道的rgb图像
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            nn.Tanh(),
            )
    #网络的最左侧的z和cond中的z是同一个z（文本编码信息少，主干是z，从z开始放大扩充，中途经过放射模块去深度融合cond，即使用cond去纠正z放大扩充的信息---仿射变化，放缩位移）
    def forward(self, noise, c): # x=noise, c=ent_emb
        # concat noise and sentence
        out = self.fc(noise)
        # noise.size(0)为batch_size
        out = out.view(noise.size(0), 8*self.ngf, 4, 4)
        cond = torch.cat((noise, c), dim=1)
        # fuse text and visual features
        for GBlock in self.GBlocks:
            out = GBlock(x=out, y=c)


            #最后输出batch_size *（32*1）* 256 * 256
        # convert to RGB image
        out = self.to_rgb(out)
        return out

# overlayAffine with skip-z&trunc
class NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
        super(NetG, self).__init__()
        self.cond_dim = cond_dim
        self.ngf = ngf  # 生成器feature map数(计算通道数量的一倍增量值)
        # input noise (batch_size, 100)
        # fc层输出的结果进入第一个size为4的df层（（ngf*8）*（4*4））------ngf*2**n为不同df层的通道扩充
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        # self.AttnBlocks = nn.ModuleList([])
        # [i，o]：[8,8][8,8][8,8][8,4][4,2][2,1]
        # channel_nums：[nf * i，nf * o]
        in_out_pairs = get_G_in_out_chs(ngf, imsize)
        former_dim = 0
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.GBlocks.append(G_Block(cond_dim+100, in_ch, out_ch, upsample=True, former_dim=former_dim))
            former_dim = out_ch
            # self.AttnBlocks.append(GlobalAttentionGeneral(in_ch, cond_dim))

        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            # 最后一个是32 * 1的out_ch，通过卷积提取特征生成3个通道的rgb图像
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            nn.Tanh(),
        )

    # 网络的最左侧的z和cond中的z是同一个z（文本编码信息少，主干是z，从z开始放大扩充，中途经过放射模块去深度融合cond，即使用cond去纠正z放大扩充的信息---仿射变化，放缩位移）
    def forward(self, noise, s):  # x=noise, c=ent_emb

        # concat noise and sentence
        cond = torch.cat((noise, s), dim=1)
        out = self.fc(noise)
        # noise.size(0)为batch_size
        out = out.view(noise.size(0), 8 * self.ngf, 4, 4)
        overlay = None
        # fuse text and visual features
        for i, GBlock in enumerate(self.GBlocks):
            
            out, overlay = GBlock(x=out, y=cond, overlay=overlay)

            # 最后输出batch_size *（32*1）* 256 * 256
        # convert to RGB image
        out = self.to_rgb(out)
        return out





# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf, imsize=128, ch_size=3):
        super(NetD, self).__init__()
        #size不变，扩充通道数量
        self.conv_img = nn.Conv2d(ch_size, ndf, 3, 1, 1)
        # build DBlocks
        self.DBlocks = nn.ModuleList([])
        # [1,2][2,4][4,8][8,8][8,8][8,8]
        in_out_pairs = get_D_in_out_chs(ndf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.DBlocks.append(D_Block(in_ch, out_ch))


    def forward(self,x):
        #先将3通道转换为第一个DownBlock所需的通道数ndf * 1（32*1）
        out = self.conv_img(x)

        for DBlock in self.DBlocks:
            out = DBlock(out)
            # 最后输出batch_size *（32*8）* 4 * 4
            # if out.shape[3]==16:
            #     localF = out
            # #     #取256 * 16 * 16 作为局部特征

        globalF = out

        return globalF#, localF
#原始的NetC
class _NetC(nn.Module):
    def __init__(self, ndf, cond_dim=256):
        super(_NetC, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf*8+cond_dim, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False),
        )
       

    def forward(self, out, y):
        y = y.view(-1, self.cond_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out



class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample, former_dim):
        super(G_Block, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        #一个upblock由两个融合块组成

        self.fuse1 = DFBLK(cond_dim, in_ch, former_dim)
        self.fuse2 = DFBLK(cond_dim, out_ch, in_ch)

        if self.learnable_sc:
            #保持size不变，通道数量变化，使用1*1卷积核
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)


    #如果输入输出的通道数量不一致，那么通过卷积层进行剪切提取（大到小），因为最后输出时需要再融合输入x
    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x
    #一个df模块由两个放射模块组成，第一个放射模块会通过卷积缩小channel_size到out_size，第二个不变，但是他们的大小都没变，都使用了padding
    def residual(self, h, y_z=None, y=None, y_w=None, attn=None, mask=None, overlay=None):
        
        h, wb = self.fuse1(x=h, y=y, overlay=overlay)
        h = self.c1(h)
       
        h, wb = self.fuse2(x=h, y=y, overlay=wb)
        h = self.c2(h)
        return h, wb

    def forward(self, x, y_z=None, y=None, y_w=None, attn=None, mask=None, overlay=None):
        #进行深度融合前的上采样层（scale_factor=2表示按照两倍放大）：4*4变8*8变16*16变32*32。。。变256*256
        if self.upsample==True:
            x = F.interpolate(x, scale_factor=2)

        
        res, wb = self.residual(x, y=y, overlay=overlay)
        return self.shortcut(x) + res, wb




class D_Block(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            #第一个卷积使得size减半，且通道增加
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        #torch.zeros(1) --> tensor([0.])
        #nn.Parameter()将这个设置为可训练的参数
        self.gamma = nn.Parameter(torch.zeros(1))
        

    def forward(self, x):
        
        #残差块变化了通道数量和size
        res = self.conv_r(x)
       
        #为了融合残差块，x需要变换通道和size
        if self.learned_shortcut:
            #size不变，通道增加
            x = self.conv_s(x)
        if self.downsample:
            #按照2倍缩小进行下采样
            x = F.avg_pool2d(x, 2)
       
        return x + self.gamma*res



#深度融合模块
class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch, former_dim):
        super(DFBLK, self).__init__()
        
        self.affine0 = _Affine_(cond_dim, in_ch, former_dim)
        self.affine1 = _Affine_(cond_dim, in_ch, in_ch)
        

    def forward(self, x, y=None, y_w=None, attn=None, mask=None, overlay=None):
        
        h, wb = self.affine0(x=x, y=y, overlay=overlay)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h, wb = self.affine1(x=h, y=y, overlay=wb)
        h = nn.LeakyReLU(0.2, inplace=True)(h)

        return h, wb



class _Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(_Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)
        # [] --> [[]] m * n
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        # 权重和偏置的维度对应着channel_size，需要扩充维度以供进行矩阵运算
        size = x.size()
        # [[[[]]]] batch_size * channel_size * w * d
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class _Affine_(nn.Module):
    def __init__(self, cond_dim, num_features, former_dim):
        super(_Affine_, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim+former_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))

        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim+former_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))

        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None, overlay=None):
        if overlay is not None:
            overlay_w = overlay[0:(overlay.shape[0])//2]
            overlay_b = overlay[(overlay.shape[0])//2:]

            weight = self.fc_gamma(torch.cat((y, overlay_w), dim=1))
            bias = self.fc_beta(torch.cat((y, overlay_b), dim=1))
        else:
            weight = self.fc_gamma(y)
            bias = self.fc_beta(y)
        #(batch_size*2) * dim
        wb = torch.cat((weight, bias), dim=0)
        # [] --> [[]] m * n
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        # 权重和偏置的维度对应着channel_size，需要扩充维度以供进行矩阵运算
        size = x.size()
        # [[[[]]]] batch_size * channel_size * w * d
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias, wb



def get_G_in_out_chs(nf, imsize):

    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    #[8,8][8,8][8,8][8,4][4,2][2,1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    #[1,2][2,4][4,8][8,8][8,8][8,8]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs
