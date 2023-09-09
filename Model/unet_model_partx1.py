""" Full assembly of the parts to form the complete network """

from unet_parts_sg_new import *
from model_new_unet_sg_new1 import *

class HIBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(HIBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.norm = nn.InstanceNorm2d(out_size//2, affine=True)


    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out

class INBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(INBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.norm = nn.InstanceNorm2d(out_size, affine=True)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.norm(out)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out


class BABlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(BABlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.norm = nn.BatchNorm2d(out_size, affine=True)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.norm(out)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out

class HIDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            HIBlock(in_channels, out_channels,0)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class INDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            INBlock(in_channels, out_channels,0)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class BADown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BABlock(in_channels, out_channels,0)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(ResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.batch1 = nn.BatchNorm2d(out_size)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.batch2 = nn.BatchNorm2d(out_size)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.conv_1(x)
        out= self.batch1(out)
        out = self.relu_1(out)
        out=self.conv_2(out)
        out = self.batch2(out)
        out = self.relu_2(out)
        out += self.identity(x)
        return out


class NORBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(NORBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.batch1 = nn.BatchNorm2d(out_size)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.batch2 = nn.BatchNorm2d(out_size)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.conv_1(x)
        out= self.batch1(out)
        out = self.relu_1(out)
        out=self.conv_2(out)
        out = self.batch2(out)
        out = self.relu_2(out)
        return out

class ResUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResBlock(in_channels, out_channels, 0)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResBlock(in_channels, out_channels, 0)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class NORUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = NORBlock(in_channels, out_channels, 0)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = NORBlock(in_channels, out_channels, 0)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, dilation=1, act='prelu'):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=dilation, bias=True,
                              dilation=dilation)
        self.batch= nn.BatchNorm2d(growthRate)

        if act == 'prelu':
            self.act = nn.PReLU(growthRate)
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        # (kernel_size - 1) // 2 + 1

        out = self.act(self.batch(self.conv(x)))
        out = torch.cat((x, out), 1)
        return out


# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, dilation, act='***'):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate, dilation=dilation, act=act))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


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

        self.inc = HIBlock(self.n_channels, 64,0)
        self.sgin = SG_IN(2,32)
        self.down1 = HIDown(64+32, 128)
        self.sgdown1 = SG_DOWN(32, 64)
        self.down2 = HIDown(128+64, 256)
        self.sgdown2 = SG_DOWN(64, 128)
        self.down3 = HIDown(256+128, 512)
        self.sgdown3 = SG_DOWN(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = HIDown(512+256, 1024 // factor)
        self.sgdown4 = SG_DOWN(256, 512// factor)

        self.skip_DRDB=DRDB(512+256,3,64,3)

        self.up1 = ResUp(1024+512, (512+256) // factor, bilinear)
        self.skip_DRDB1 = DRDB(512+256, 3, 64, 3)
        self.up2 = ResUp(512+256, (256+128) // factor, bilinear)
        self.skip_DRDB2 = DRDB(256 + 128, 3, 32, 3)
        self.up3 = ResUp(256+128, (128+64) // factor, bilinear)
        self.skip_DRDB3 = DRDB(128+64, 3, 16, 3)
        self.up4 = ResUp(128+64, 64+32, bilinear)
        self.skip_DRDB4 = DRDB(64+32, 3, 8, 3)
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
        x5=self.skip_DRDB(x5)
        x4 = self.skip_DRDB1(x4)
        x3 = self.skip_DRDB2(x3)
        x2 = self.skip_DRDB3(x2)
        x1 = self.skip_DRDB4(x1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits=self.tanh(logits)
        return logits

class UNet_PARTX1_1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_PARTX1_1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = HIBlock(self.n_channels, 64,0)
        self.down1 = HIDown(64, 128)
        self.down2 = HIDown(128, 256)
        self.down3 = HIDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = HIDown(512, 1024 // factor)

        self.up1 = ResUp(1024, 512// factor, bilinear)
        self.up2 = ResUp(512, 256 // factor, bilinear)
        self.up3 = ResUp(256, 128 // factor, bilinear)
        self.up4 = ResUp(128 , 64 //factor, bilinear)
        self.outc = OutConv(32, n_classes)
        self.tanh=nn.Tanh()

    def forward(self, x):
        g=x
        x1 = self.inc(g)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits=self.tanh(logits)
        return logits

class UNet_PARTX1_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_PARTX1_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = HIBlock(self.n_channels, 64,0)
        self.sgin = SG_IN(2,32)
        self.down1 = HIDown(64+32, 128)
        self.sgdown1 = SG_DOWN(32, 64)
        self.down2 = HIDown(128+64, 256)
        self.sgdown2 = SG_DOWN(64, 128)
        self.down3 = HIDown(256+128, 512)
        self.sgdown3 = SG_DOWN(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = HIDown(512+256, 1024 // factor)
        self.sgdown4 = SG_DOWN(256, 512// factor)


        self.up1 = ResUp(1024+512, (512+256) // factor, bilinear)
        self.up2 = ResUp(512+256, (256+128) // factor, bilinear)
        self.up3 = ResUp(256+128, (128+64) // factor, bilinear)
        self.up4 = ResUp(128+64, 64+32, bilinear)
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

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits=self.tanh(logits)
        return logits

class UNet_PARTX1_3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_PARTX1_3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = HIBlock(self.n_channels, 64,0)
        self.down1 = HIDown(64, 128)
        self.down2 = HIDown(128, 256)
        self.down3 = HIDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = HIDown(512, 1024 // factor)

        self.skip_DRDB=DRDB(512,3,64,3)
        self.up1 = ResUp(1024, (512) // factor, bilinear)
        self.skip_DRDB1 = DRDB(512, 3, 64, 3)
        self.up2 = ResUp(512, (256) // factor, bilinear)
        self.skip_DRDB2 = DRDB(256 , 3, 32, 3)
        self.up3 = ResUp(256, (128) // factor, bilinear)
        self.skip_DRDB3 = DRDB(128, 3, 16, 3)
        self.up4 = ResUp(128, 64//factor, bilinear)
        self.skip_DRDB4 = DRDB(64, 3, 8, 3)
        self.outc = OutConv(32, n_classes)
        self.tanh=nn.Tanh()

    def forward(self, x):
        dmap = x.clone()
        r = dmap[:, 0:1, :, :]
        g = dmap
        b = dmap[:, 2:3, :, :]
        rb=torch.cat([r,b],1)
        x1 = self.inc(g)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
        x5=self.skip_DRDB(x5)
        x4 = self.skip_DRDB1(x4)
        x3 = self.skip_DRDB2(x3)
        x2 = self.skip_DRDB3(x2)
        x1 = self.skip_DRDB4(x1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits=self.tanh(logits)
        return logits
if __name__ == '__main__':
    from thop import profile,clever_format
    unet=UNet_PARTX1_3(4,3)
    input1=torch.randn((1,4,720,720))

    out=unet(input1)
    print(out.shape)
    flops,param=profile(unet,inputs=(input1,))

    print(flops/1e9,param/1e6)