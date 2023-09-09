""" Full assembly of the parts to form the complete network """

from unet_parts_ba import *


class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes,hc, bilinear=True):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, hc)
        self.down1 = Down(hc, hc*2)
        self.down2 = Down(hc*2, hc*4)
        self.down3 = Down(hc*4, hc*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(hc*8, hc*16 // factor)
        self.up1 = Up(hc*16, hc*8 // factor, bilinear)
        self.up2 = Up(hc*8, hc*4 // factor, bilinear)
        self.up3 = Up(hc*4, hc*2 // factor, bilinear)
        self.up4 = Up(hc*2, hc, bilinear)
        self.outc = OutConv(hc, n_classes)



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes,hc, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, hc)
        self.down1 = Down(hc, hc*2)
        self.down2 = Down(hc*2, hc*4)
        factor = 2 if bilinear else 1
        self.down3 = Down(hc*4, hc*8//factor)
        self.up2 = Up(hc*8, hc*4 // factor, bilinear)
        self.up3 = Up(hc*4, hc*2 // factor, bilinear)
        self.up4 = Up(hc*2, hc, bilinear)
        self.outc = OutConv(hc, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet(nn.Module):
    def __init__(self, ):
        super(UNet, self).__init__()
        self.unet1=UNet1(3,3,24)
        self.up1 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
        self.unet2 = UNet2(3, 3,24)
        self.up2 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
        self.unet3 = UNet2(3, 3,24)
        self.up3 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
        self.unet4 = UNet2(3, 3,16)
        self.tanh = nn.Tanh()

    def forward(self, x1,x2,x3,x4):
        x1=self.unet1(x1)
        x1=self.up1(x1)
        x12=x1+x2
        x2 = self.unet2(x12)
        x2=x12+x2
        x2 = self.up2(x2)
        x23 = x2 + x3
        x3 = self.unet3(x23)
        x3 = x23 + x3
        x3 = self.up3(x3)
        x34 = x3+x4
        x4 = self.unet4(x34)
        logits=x4+x34
        return x1,x2,x3,self.tanh(logits)

if __name__ == '__main__':
    unet=UNet()
    input1=torch.randn((2,3,320,320))
    input2 = torch.randn((2, 3, 160, 160))
    input3 = torch.randn((2, 3, 80, 80))
    input4 = torch.randn((2, 3, 40, 40))
    _,_,_,out=unet(input4,input3,input2,input1)
    print(out.shape)
