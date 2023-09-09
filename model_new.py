from collections import OrderedDict
import torch.nn as nn
from color_loss import *
import torchvision

#构造模型序列模块
def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

#conv+relu
def conv_relu(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
    CR = []
    CR.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    CR.append(nn.ReLU(inplace=True))
    return sequential(*CR)

#conv+BatchNorm+relu
def convb_relu(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
    CBR = []
    CBR.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    CBR.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
    CBR.append(nn.ReLU(inplace=True))
    return sequential(*CBR)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


#全卷积
class Relight1(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, bias = True):
        super(Relight1, self).__init__()

        m_head = conv_relu(in_nc, nc, bias=bias)
        m_body=[]
        for i in range(18):
            m_body.append(convb_relu(nc, nc, bias=bias))

        m_tail = nn.Conv2d(nc,out_nc, 3, 1, 1, bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        #n为残差
        n = self.model(x)
        #输出为输入减去残差即为想要的图
        return x-n



class UpConv(nn.Module):
    def __init__(self,in_nc=1, out_nc=1):
        super(UpConv, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.body(x)


#反卷积
class Relight2(nn.Module):
    def __init__(self, in_nc=1, out_nc=1):
        super(Relight2, self).__init__()

        # 初始卷积块
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # 下采样
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # 残差网络块
        for _ in range(6):
            model += [ResidualBlock(in_features)]

        # 上采样
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # 输出层
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, out_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

#上采样
class Relight3(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(Relight3, self).__init__()

        # 初始卷积块
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # 下采样
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # 残差网络块
        for _ in range(6):
            model += [ResidualBlock(in_features)]

        # 上采样
        out_features = in_features // 2
        for _ in range(2):
            model += [UpConv(in_features, out_features*4)]
            in_features = out_features
            out_features = in_features // 2

        # 输出层
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, out_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


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
        fake = G(input)
        loss_G_FM += self.FMcriterion(fake, target)
        loss_G += loss_G_FM  * self.opt.lambda_FM

        fake1=self.adjust_dynamic_range(fake.cpu().detach().float().numpy(), drange_in=[-1., 1.]
                                             , drange_out=[0, 255])
        target1=self.adjust_dynamic_range(target.cpu().detach().float().numpy(), drange_in=[-1., 1.]
                                             , drange_out=[0, 255])
        if self.opt.VGG_loss:
            loss_G_VGG_FM = 0
            weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            real_features_VGG, fake_features_VGG = self.VGGNet(target), self.VGGNet(fake)

            for i in range(len(real_features_VGG)):
                loss_G_VGG_FM += weights[i] * self.FMcriterion(fake_features_VGG[i], real_features_VGG[i])
            loss_G += loss_G_VGG_FM * self.opt.lambda_FM

        fake1=torch.from_numpy(fake1).to(self.device)
        target1=torch.from_numpy(target1).to(self.device)
        color_loss=self.cl(self.blur_rgb(fake1),self.blur_rgb(target1))
        loss_G+=color_loss
        grad_loss=gradient(fake1,target1,self.gradnet)
        loss_G+=grad_loss
        return loss_G,target, fake


if __name__ == '__main__':
    import torch
    model = Relight3(3)
    input = torch.randn(1, 3, 360, 540)
    print(model(input).shape)

