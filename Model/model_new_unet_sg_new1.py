import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from layers import Attention
import math
from color_loss import *
import torchvision
import common_new as common
from unet_model_origin import UNet



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        # self.batchnormal = nn.InstanceNorm2d(out_channels)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        # x=self.batchnormal(x)
        # x=self.lrelu(x)
        return x

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

class UpConv(nn.Module):
    def __init__(self,in_nc):
        super(UpConv, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_nc, in_nc*4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.body(x)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,bias=True):
        super(Conv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding=1,bias=bias)
        self.batchnormal = nn.InstanceNorm2d(out_channels)


    def forward(self, x):
        x = self.conv1(x)
        x=self.batchnormal(x)
        x=self.lrelu(x)
        return x

    def lrelu(self, x):
        # outt = torch.max(0.2 * x, x)
        outt = F.relu(x)
        return outt

class Conv2d_norelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,bias=True):
        super(Conv2d_norelu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding=1,bias=bias)

    def forward(self, x):
        x = self.conv1(x)

        return x



class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.InstanceNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.InstanceNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.InstanceNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.InstanceNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        return out


class SA_NET(nn.Module):
    def __init__(self, inchannel):
        super(SA_NET, self).__init__()
        self.conv1 = SeparableConv2d(inchannel,16,3,2)
        self.batchnormal = nn.InstanceNorm2d(16)

        self.maxpool = nn.AvgPool2d(2, 2)
        self.conv1_1 = SeparableConv2d(16,32,3,2)
        self.attention=Attention(32)
        self.upsample_0=UpConv(32)
        self.conv2_0 = SeparableConv2d(32, 32, 3, 1)
        self.upsample_1 = UpConv(32)
        self.conv2_1 = SeparableConv2d(32, 32, 3, 1)
        self.upsample_2 = UpConv(32)
        self.conv2_2 = SeparableConv2d(32, 16, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnormal(x)
        x = self.lrelu(x)
        x = self.maxpool(x)
        x = self.conv1_1(x)
        x = self.attention(x)
        x = self.upsample_0(x)
        x = self.conv2_0(x)
        x = self.upsample_1(x)
        x = self.conv2_1(x)
        x = self.upsample_2(x)
        x = self.conv2_2(x)
        return x

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(4, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        # self.maxpool = nn.AvgPool2d(2, 2)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.trans3=Transition(nChannels, 1)

        self.bn1 = nn.InstanceNorm2d(nChannels)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        # out=self.maxpool(out)
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        # out=self.trans3(out)
        return out


class Generator1(nn.Module):
    def __init__(self,):
        super(Generator1, self).__init__()
        self.sannetr = SA_NET(1)
        self.sannetg = SA_NET(4)
        self.sannetb = SA_NET(1)
        self.dense=DenseNet(4,16,0.5,True)
        self.final = Conv2d(48,48)
        self.tanh=nn.Tanh()


    def forward(self, x):
        dmap = x.clone()
        r =dmap[:,0:1, :, :]
        g=dmap
        b = dmap[:,2:3, :, :]

        r_out=self.sannetr(r)
        g_out1=self.sannetg(g)
        g_out2=self.dense(g)
        b_out=self.sannetb(b)
        g_out=g_out1+g_out2

        x=torch.cat([r_out,g_out,b_out],dim=1)
        x0=self.final(x)
        return x0

class Generator2(nn.Module):
    def __init__(self,):
        super(Generator2, self).__init__()
        self.srnet=UNet(4,3)
    def forward(self, x0):
        out=self.srnet(x0)
        return out

class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()
        self.stage2=Generator2()
    def forward(self, x0):
        out = self.stage2(x0)
        return out


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg_pretrained_layers = torchvision.models.vgg19(pretrained=True).features
        self.block_1 = nn.Sequential()
        self.block_2 = nn.Sequential()
        self.block_3 = nn.Sequential()
        self.block_4 = nn.Sequential()
        self.block_5 = nn.Sequential()

        for i in range(2):
            self.block_1.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(2, 7):
            self.block_2.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(7, 12):
            self.block_3.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(12, 21):
            self.block_4.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(21, 30):
            self.block_5.add_module(str(i), vgg_pretrained_layers[i])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out_1 = self.block_1(x)
        out_2 = self.block_2(out_1)
        out_3 = self.block_3(out_2)
        out_4 = self.block_4(out_3)
        out_5 = self.block_5(out_4)
        out = [out_1, out_2, out_3, out_4, out_5]

        return out



class Loss(object):
    def __init__(self, opt,device):
        self.opt = opt
        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.blur_rgb = Blur(3,device)
        self.cl= ColorLoss()
        self.gradnet=Gradient_Net(device)

        if opt.VGG_loss:
            self.VGGNet = VGG19()
            if opt.gpu_ids != -1:
                self.VGGNet = self.VGGNet.to(device)
        self.device = device

    @staticmethod
    def adjust_dynamic_range(data, drange_in, drange_out):
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    def __call__(self, G, input, target):
        loss_G = 0
        loss_G_FM = 0
        lr1 = G(input)
        loss_G_FM += self.FMcriterion(lr1, target)
        loss_G += loss_G_FM
        #
        fake1=self.adjust_dynamic_range(lr1.cpu().detach().float().numpy(), drange_in=[-1., 1.]
                                             , drange_out=[0, 255])
        target1=self.adjust_dynamic_range(target.cpu().detach().float().numpy(), drange_in=[-1., 1.]
                                             , drange_out=[0, 255])
        if self.opt.VGG_loss:
            loss_G_VGG_FM = 0
            weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            real_features_VGG, fake_features_VGG = self.VGGNet(target), self.VGGNet(lr1)

            for i in range(len(real_features_VGG)):
                loss_G_VGG_FM += weights[i] * self.FMcriterion(fake_features_VGG[i], real_features_VGG[i])
            loss_G += loss_G_VGG_FM * self.opt.lambda_FM

        fake1=torch.from_numpy(fake1).to(self.device)
        target1=torch.from_numpy(target1).to(self.device)
        color_loss=self.cl(self.blur_rgb(fake1),self.blur_rgb(target1))
        loss_G+=color_loss
        grad_loss=gradient(fake1,target1,self.gradnet)
        loss_G+=grad_loss
        return loss_G,target,lr1

if __name__ == '__main__':
    input=torch.randn(2,4, 360,360)
    model=Generator()
    out1=model(input)
    print(out1.shape)

