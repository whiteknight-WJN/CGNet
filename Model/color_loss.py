import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


class Blur(nn.Module):
    def __init__(self, nc,device):
        super(Blur, self).__init__()
        self.nc = nc
        kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))
        x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
        return x


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x1, x2):
        return torch.mean(torch.pow((x1 - x2), 2)).div(2 * x1.size()[0])


class Gradient_Net(nn.Module):
  def __init__(self,device):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x)
    grad_y = F.conv2d(x, self.weight_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    return gradient


def gradient(x,y,grad_net):
    x=0.299*x[:,0:1,:,:]+0.587*x[:,1:2,:,:]+0.114*x[:,2:3,:,:]
    y = 0.299 * y[:, 0:1, :, :] + 0.587 * y[:, 1:2, :, :] + 0.114 * y[:, 2:3, :, :]
    g1 = grad_net(x)
    g2 = grad_net(y)
    gloss=F.l1_loss(g1,g2)
    return gloss


if __name__ == '__main__':
    cl = ColorLoss()

    # rgb example
    blur_rgb = Blur(3)
    img_rgb1 = torch.randn(4, 3, 40, 40)
    img_rgb2 = torch.randn(4, 3, 40, 40)
    blur_rgb1 = blur_rgb(img_rgb1)
    blur_rgb2 = blur_rgb(img_rgb2)
    print(cl(blur_rgb1, blur_rgb2))

    # gray example
    blur_gray = Blur(1)
    img_gray1 = torch.randn(4, 1, 40, 40)
    img_gray2 = torch.randn(4, 1, 40, 40)
    blur_gray1 = blur_gray(img_gray1)
    blur_gray2 = blur_gray(img_gray2)
    print(cl(blur_gray1, blur_gray2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
