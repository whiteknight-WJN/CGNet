import time

import torch
from model_new_unet_msec_raw import Generator
from model_new import Relight1
from option import TrainOption, TestOption
from utils import Manager, update_lr, weights_init
import datetime
import os
from tqdm import tqdm
from raw_rgb import *
import numpy as np
from torchvision import transforms
import cv2
from niqe import niqe
import utils_image as util
import PIL.Image as Image
from tqdm import tqdm



output_dir = r'./result_msec_raw_real'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
device=torch.device('cpu')
G = Generator().to(device)
G_static=torch.load('./checkpoints/aia2ha_new_unet_msec_raw/Model/height_512/480_G.pt')
model_static={k.replace('module.',''):v for k,v in G_static.items()}
# model_static.update(G_static)
G.load_state_dict(model_static)

G.eval()


source_path='../../../../../data4/hongyang/REAL_DATASET_NEW/'
testfile=open('../test_real.txt','r')
testfile=testfile.readlines()

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def tensor2image(image_tensor):
    np_image = image_tensor.squeeze().cpu().float().numpy()

    if len(np_image.shape) == 3:
        np_image = np.transpose(np_image, (1, 2, 0))  # HWC
    else:
        pass

    np_image = adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
    np_image = np.clip(np_image, 0, 255).astype(np.uint8)
    return np_image


preprocessing = transforms.Compose([
            transforms.ToTensor()])
list1=[3,5,8,10]
rata=[2.05,1.7,1.4,1.2]
for i1,r in zip(list1,rata):
    print(i1,r)
    psnr_value = []
    ssim_value = []
    niqe_value = []
    for i in tqdm(testfile):
        t1=time.time()
        imgfile=source_path+i.split()[0]+'/%d.png'%i1
        gtfile=source_path+i.split()[0]+'/gt.png'
        x_data=cv2.imread(imgfile,-1)


        h,w=x_data.shape[0],x_data.shape[1]
        gt_img = cv2.imread(gtfile, -1)
        k1=min(h,w)
        hh=h//2
        wh=w//2

        if k1>720:
            k=360
        else:
            k=256

        x_data=x_data[hh-k:hh+k,wh-k:wh+k]
        x_data1 = cv2.pyrDown(x_data)
        x_data2 = cv2.pyrDown(x_data1)
        x_data3 = cv2.pyrDown(x_data2)
        gt_img = gt_img[hh - k:hh + k, wh - k:wh + k]
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
        x_data=(x_data-32767.5)/32767.5
        x_data1 = (x_data1 - 32767.5)/32767.5
        x_data2 = (x_data2 - 32767.5)/32767.5
        x_data3 = (x_data3 - 32767.5)/32767.5
        x=preprocessing(x_data).float().unsqueeze(0).to(device)
        x1 = preprocessing(x_data1).float().unsqueeze(0).to(device)
        x2 = preprocessing(x_data2).float().unsqueeze(0).to(device)
        x3 = preprocessing(x_data3).float().unsqueeze(0).to(device)
        with torch.no_grad():
            _,_,_,y=G(x3,x2,x1,x)
            out_y=tensor2image(y.cpu())
            out_y = cv2.cvtColor(out_y, cv2.COLOR_RGB2BGR)
            out_y = np.clip(out_y * r, 0, 255)
            out_y = np.uint8(out_y)
            out=np.hstack((out_y,gt_img))
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            t2=time.time()
            print(t2-t1)
    #         psnr_value.append(util.calculate_psnr(out_y,gt_img,border=0))
    #         ssim_value.append(util.calculate_ssim(out_y,gt_img,border=0))
    #         niqe_value.append(niqe(cv2.cvtColor(out_y,cv2.COLOR_BGR2GRAY)))
    #         savepath1 = output_dir + '/' + '%s_%d_predict.png'%(i,i1)
    #         cv2.imencode('.png', out)[1].tofile(savepath1)
    #
    # psnr_value=np.array(psnr_value)
    # ssim_value=np.array(ssim_value)
    # niqe_value=np.array(niqe_value)
    # print('PSNR=%.4f,SSIM=%.4f,NIQE=%.4f'%(np.mean(psnr_value),np.mean(ssim_value),np.mean(niqe_value)))
    #
    #

