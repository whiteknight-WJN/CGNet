import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os
from torchvision import transforms
import cv2
from PIL import Image


def rand_crop(data, label):
    size=480
    width1 = random.randint(0, data.shape[1] - size)
    height1 = random.randint(0, data.shape[0] - size)
    width2 = width1 + size
    height2 = height1 + size

    data = data[height1:height2,width1:width2 ]
    label = label[height1:height2,width1:width2 ]

    return data, label


class MyDataset(Dataset):
    def __init__(self,filename):
        self.name=filename
        self.frame_list = list()
        for line in os.listdir(self.name):
            self.frame_list.append(self.name+'/'+line)
        self.len = len(self.frame_list)
        self.preprocessing = transforms.Compose([
            transforms.ToTensor()])
    def getframe(self,index):
        path = self.frame_list[index]
        x_data = Image.open(path + '/1.png')
        y_data = Image.open(path + '/gt.png').convert('RGB')
        x_data, y_data = rand_crop(x_data, y_data)
        x_data = np.asarray(x_data)
        rate=random.uniform(0.0,12.0)
        x_data=np.clip(x_data*rate,0,255)

        x_data=np.uint8(x_data)
        x_data=(x_data-127.5)/127.5
        # x_data=cv2.resize(x_data,(540,320))

        y_data = np.asarray(y_data)
        y_data = (y_data - 127.5) / 127.5


        # y_data=cv2.resize(y_data,(540,320))
        # x,y=self._get_random_crop(x_data,y_data)
        return x_data,y_data

    def __getitem__(self, index):

        x_data,y_data=self.getframe(index)
        self.x_data = self.preprocessing(x_data).float()
        self.y_data = self.preprocessing(y_data).float()

        return self.x_data,self.y_data
    def __len__(self):
        return self.len

class MyDataset1(Dataset):
    def __init__(self,filename,ftxt):
        self.name=filename
        self.frame_list = list()
        rate=['3.png','5.png','8.png','10.png']
        f=open(ftxt,'r')
        fdata=f.readlines()
        for line in fdata:
            line1=line.split()[0]
            for k in rate:
                self.frame_list.append([self.name+'/'+line1+'/'+k,self.name+'/'+line1+'/gt.png'])
        self.len = len(self.frame_list)
        self.preprocessing = transforms.Compose([
            transforms.ToTensor()])
    def getframe(self,index):
        path = self.frame_list[index]


        x_data = cv2.imread(path[0],-1)
        y_data = cv2.imread(path[1],-1)
        y_data=cv2.cvtColor(y_data,cv2.COLOR_BGR2RGB)

        x_data, y_data = rand_crop(x_data, y_data)

        x_data=np.asarray(x_data)

        x_data=(x_data-32767.5)/32767.5
        # x_data=cv2.resize(x_data,(540,320))

        y_data = np.asarray(y_data)

        y_data = (y_data - 127.5) / 127.5

        # y_data=cv2.resize(y_data,(540,320))
        # x,y=self._get_random_crop(x_data,y_data)
        return x_data,y_data

    def __getitem__(self, index):

        x_data,y_data=self.getframe(index)
        self.x_data = self.preprocessing(x_data).float()
        self.y_data = self.preprocessing(y_data).float()

        return self.x_data,self.y_data
    def __len__(self):
        return self.len

class MyDataset2(Dataset):
    def __init__(self,filename,ftxt):
        self.name=filename
        self.frame_list = list()
        rate=['8.png']
        f=open(ftxt,'r')
        fdata=f.readlines()
        for line in fdata:
            line1=line.split()[0]
            for k in rate:
                self.frame_list.append([self.name+'/'+line1+'/'+k,self.name+'/'+line1+'/gt.png'])
        self.len = len(self.frame_list)
        self.preprocessing = transforms.Compose([
            transforms.ToTensor()])
    def getframe(self,index):
        path = self.frame_list[index]


        x_data = cv2.imread(path[0],-1)
        y_data = cv2.imread(path[1],-1)
        y_data=cv2.cvtColor(y_data,cv2.COLOR_BGR2RGB)

        x_data, y_data = rand_crop(x_data, y_data)

        x_data=np.asarray(x_data)

        x_data=(x_data-32767.5)/32767.5
        # x_data=cv2.resize(x_data,(540,320))

        y_data = np.asarray(y_data)

        y_data = (y_data - 127.5) / 127.5

        # y_data=cv2.resize(y_data,(540,320))
        # x,y=self._get_random_crop(x_data,y_data)
        return x_data,y_data

    def __getitem__(self, index):

        x_data,y_data=self.getframe(index)
        self.x_data = self.preprocessing(x_data).float()
        self.y_data = self.preprocessing(y_data).float()

        return self.x_data,self.y_data
    def __len__(self):
        return self.len


class MyDataset3(Dataset):
    def __init__(self,filename,ftxt):
        self.name=filename
        self.frame_list = list()
        rate=['rgb3.png','rgb5.png','rgb8.png','rgb10.png']
        f=open(ftxt,'r')
        fdata=f.readlines()
        for line in fdata:
            line1=line.split()[0]
            for k in rate:
                self.frame_list.append([self.name+'/'+line1+'/'+k,self.name+'/'+line1+'/gt.png'])
        self.len = len(self.frame_list)
        self.preprocessing = transforms.Compose([
            transforms.ToTensor()])
    def getframe(self,index):
        path = self.frame_list[index]


        x_data = cv2.imread(path[0],-1)
        y_data = cv2.imread(path[1],-1)
        y_data=cv2.cvtColor(y_data,cv2.COLOR_BGR2RGB)

        x_data, y_data = rand_crop(x_data, y_data)

        x_data=np.asarray(x_data)

        x_data=(x_data-127.5)/127.5
        # x_data=cv2.resize(x_data,(540,320))

        y_data = np.asarray(y_data)

        y_data = (y_data - 127.5) / 127.5

        # y_data=cv2.resize(y_data,(540,320))
        # x,y=self._get_random_crop(x_data,y_data)
        return x_data,y_data

    def __getitem__(self, index):

        x_data,y_data=self.getframe(index)
        self.x_data = self.preprocessing(x_data).float()
        self.y_data = self.preprocessing(y_data).float()

        return self.x_data,self.y_data
    def __len__(self):
        return self.len

class MyDataset4(Dataset):
    def __init__(self,filename,ftxt):
        self.name=filename
        self.frame_list = list()
        rate=['rgb3.png','rgb5.png','rgb8.png','rgb10.png']
        f=open(ftxt,'r')
        fdata=f.readlines()
        for line in fdata:
            line1=line.split()[0]
            for k in rate:
                self.frame_list.append([self.name+'/'+line1+'/'+k,self.name+'/'+line1+'/gt.png'])
        self.len = len(self.frame_list)
        self.preprocessing = transforms.Compose([
            transforms.ToTensor()])

    def getframe(self,index):
        path = self.frame_list[index]
        x_data = cv2.imread(path[0],-1)
        y_data = cv2.imread(path[1],-1)
        x_data, y_data = rand_crop(x_data, y_data)
        x_data1 = cv2.pyrDown(x_data)
        x_data2 = cv2.pyrDown(x_data1)
        x_data3 = cv2.pyrDown(x_data2)
        y_data1 = cv2.pyrDown(y_data)
        y_data2 = cv2.pyrDown(y_data1)
        y_data3 = cv2.pyrDown(y_data2)

        x_data=np.asarray(x_data)
        x_data=(x_data- 127.5)/127.5
        y_data = np.asarray(y_data)
        y_data = (y_data - 127.5) / 127.5


        x_data1 = np.asarray(x_data1)
        x_data1 = (x_data1 - 127.5) / 127.5
        y_data1 = np.asarray(y_data1)
        y_data1 = (y_data1 - 127.5) / 127.5

        x_data2 = np.asarray(x_data2)
        x_data2 = (x_data2 - 127.5) / 127.5
        y_data2 = np.asarray(y_data2)
        y_data2 = (y_data2 - 127.5) / 127.5

        x_data3 = np.asarray(x_data3)
        x_data3 = (x_data3 - 127.5) / 127.5
        y_data3 = np.asarray(y_data3)
        y_data3 = (y_data3 - 127.5) / 127.5
        return (x_data3,x_data2,x_data1,x_data),(y_data2,y_data1,y_data,y_data)

    def __getitem__(self, index):

        x_data,y_data=self.getframe(index)
        self.x_data = (self.preprocessing(x_data[0]).float(),self.preprocessing(x_data[1]).float(),self.preprocessing(x_data[2]).float(),self.preprocessing(x_data[3]).float())
        self.y_data = (self.preprocessing(y_data[0]).float(),self.preprocessing(y_data[1]).float(),self.preprocessing(y_data[2]).float(),self.preprocessing(y_data[3]).float())

        return self.x_data,self.y_data
    def __len__(self):
        return self.len

class MyDataset5(Dataset):
    def __init__(self,filename,ftxt):
        self.name=filename
        self.frame_list = list()
        rate=['3.png','5.png','8.png','10.png']
        f=open(ftxt,'r')
        fdata=f.readlines()
        for line in fdata:
            line1=line.split()[0]
            for k in rate:
                self.frame_list.append([self.name+'/'+line1+'/'+k,self.name+'/'+line1+'/gt.png'])
        self.len = len(self.frame_list)
        self.preprocessing = transforms.Compose([
            transforms.ToTensor()])

    def getframe(self,index):
        path = self.frame_list[index]
        x_data = cv2.imread(path[0],-1)
        y_data = cv2.imread(path[1],-1)
        x_data, y_data = rand_crop(x_data, y_data)
        x_data1 = cv2.pyrDown(x_data)
        x_data2 = cv2.pyrDown(x_data1)
        x_data3 = cv2.pyrDown(x_data2)
        y_data1 = cv2.pyrDown(y_data)
        y_data2 = cv2.pyrDown(y_data1)
        y_data3 = cv2.pyrDown(y_data2)

        x_data=np.asarray(x_data)
        x_data=(x_data-32767.5)/32767.5
        y_data = np.asarray(y_data)
        y_data = (y_data - 127.5) / 127.5


        x_data1 = np.asarray(x_data1)
        x_data1 = (x_data1 -32767.5)/32767.5
        y_data1 = np.asarray(y_data1)
        y_data1 = (y_data1 - 127.5) / 127.5

        x_data2 = np.asarray(x_data2)
        x_data2 = (x_data2 -32767.5)/32767.5
        y_data2 = np.asarray(y_data2)
        y_data2 = (y_data2 - 127.5) / 127.5

        x_data3 = np.asarray(x_data3)
        x_data3 = (x_data3 -32767.5)/32767.5
        y_data3 = np.asarray(y_data3)
        y_data3 = (y_data3 - 127.5) / 127.5
        return (x_data3,x_data2,x_data1,x_data),(y_data2,y_data1,y_data,y_data)

    def __getitem__(self, index):

        x_data,y_data=self.getframe(index)
        self.x_data = (self.preprocessing(x_data[0]).float(),self.preprocessing(x_data[1]).float(),self.preprocessing(x_data[2]).float(),self.preprocessing(x_data[3]).float())
        self.y_data = (self.preprocessing(y_data[0]).float(),self.preprocessing(y_data[1]).float(),self.preprocessing(y_data[2]).float(),self.preprocessing(y_data[3]).float())

        return self.x_data,self.y_data
    def __len__(self):
        return self.len