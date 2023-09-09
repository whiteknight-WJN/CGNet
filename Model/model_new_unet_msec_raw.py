import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from layers import Attention
import math
from color_loss import *
import torchvision
import common_new as common
from unet_model_msec import UNet_raw




class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()
        self.stage1=UNet_raw()
    def forward(self, x0,x1,x2,x3):

        x1,x2,x3,out1=self.stage1(x0,x1,x2,x3)
        return x1,x2,x3,out1



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

    def __call__(self, G, x1,x2,x3,input, y1,y2,y3,target):
        loss_G = 0
        loss_G_FM = 0
        x01,x02,x03,lr1 = G(x1,x2,x3,input)
        loss_G_FM += 8*self.FMcriterion(lr1, target)
        loss_G += loss_G_FM
        loss_G += self.FMcriterion(x01, y1)
        loss_G += 2*self.FMcriterion(x02, y2)
        loss_G += 4 * self.FMcriterion(x03, y3)

        return loss_G,target,lr1

if __name__ == '__main__':
    input=torch.randn(2,4, 720,720)
    input1 = torch.randn(2, 4, 360, 360)
    input2 = torch.randn(2, 4, 180, 180)
    input3 = torch.randn(2, 4, 90, 90)
    model=Generator()
    out0,out1,out2,out3=model(input3,input2,input1,input)
    print(out0.shape,out1.shape)

