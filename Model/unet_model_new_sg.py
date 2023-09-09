""" Full assembly of the parts to form the complete network """

from unet_parts_origin import *
from model_new_unet_sg_new1 import *

class SG_IN(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(SG_IN, self).__init__()
        self.conv1 = SeparableConv2d(inchannel,outchannel//2,3,1)
        self.batchnormal = nn.InstanceNorm2d(outchannel//2)
        self.conv1_1 = SeparableConv2d(outchannel//2,outchannel,3,1)
        self.conv2_0 = SeparableConv2d(outchannel, outchannel, 3, 1)
        self.batchnormal2 = nn.InstanceNorm2d(outchannel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnormal(x)
        x = self.lrelu(x)
        x = self.conv1_1(x)
        x = self.conv2_0(x)
        x = self.batchnormal2(x)
        x = self.lrelu(x)
        return x

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

class SG_DOWN(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(SG_DOWN, self).__init__()
        self.maxpool = nn.AvgPool2d(2, 2)
        self.conv1 = SeparableConv2d(inchannel,outchannel//2,3,1)
        self.conv1_1 = SeparableConv2d(outchannel//2,outchannel,3,1)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv1_1(x)
        return x

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

class SG_DOWNA(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(SG_DOWNA, self).__init__()
        self.maxpool = nn.AvgPool2d(2, 2)
        self.attention = Attention(inchannel)
        self.conv1 = SeparableConv2d(inchannel,outchannel//2,3,1)
        self.conv1_1 = SeparableConv2d(outchannel//2,outchannel,3,1)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.attention(x)
        x = self.conv1(x)
        x = self.conv1_1(x)
        return x

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.sgin = SG_IN(2,32)
        self.down1 = Down(64+32, 128)
        self.sgdown1 = SG_DOWN(32, 64)
        self.down2 = Down(128+64, 256)
        self.sgdown2 = SG_DOWN(64, 128)
        self.down3 = Down(256+128, 512)
        self.sgdown3 = SG_DOWN(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(512+256, 1024 // factor)
        self.sgdown4 = SG_DOWN(256, 512// factor)

        self.up1 = Up(1024+512, (512+256) // factor, bilinear)
        self.up2 = Up(512+256, (256+128) // factor, bilinear)
        self.up3 = Up(256+128, (128+64) // factor, bilinear)
        self.up4 = Up(128+64, 64+32, bilinear)
        self.outc = OutConv(64+32, n_classes)
        self.tanh=nn.Tanh()

    def forward(self, x):
        dmap = x.clone()
        r = dmap[:, 0:1, :, :]
        g = dmap
        b = dmap[:, 2:3, :, :]
        rb=torch.cat([r,b],1)

        x1 = self.inc(g)
        xsg1 = self.sgin(rb)
        x1=torch.cat([x1,xsg1],1)
        x2 = self.down1(x1)
        xsg2 = self.sgdown1(xsg1)
        x2=torch.cat([x2,xsg2],1)
        x3 = self.down2(x2)
        xsg3 = self.sgdown2(xsg2)
        x3 = torch.cat([x3, xsg3], 1)
        x4 = self.down3(x3)
        xsg4 = self.sgdown3(xsg3)
        x4 = torch.cat([x4, xsg4], 1)
        x5 = self.down4(x4)
        xsg5 = self.sgdown4(xsg4)
        x5 = torch.cat([x5, xsg5], 1)

        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits=self.tanh(logits)
        return logits

if __name__ == '__main__':
    unet=UNet(4,3)
    input1=torch.randn((2,4,160,160))
    input2 = torch.randn((2, 48, 160, 160))
    out=unet(input1)
    print(out.shape)
    # model=SG_IN(2)
    # input=torch.randn((2,2,160,160))
    # output=model(input)
    # print(output.shape)
