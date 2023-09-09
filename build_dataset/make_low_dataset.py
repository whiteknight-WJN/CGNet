import os
import shutil
import random
from tqdm import tqdm
path0='./low_light_enhance/PNLI(2000pairs)/PNLI_1200_800/'
path1='./low_high_datasets/PNLI_1/'
filelist=os.listdir(path0+'low')
random.seed(100)
random.shuffle(filelist)
length=len(filelist)
train=filelist[:1700]
test=filelist[1700:]
def copyFiles(source,target):
   shutil.copy(source,target)

for data in tqdm(train):
   sourcepath0=path0+'low/'+data
   sourcepath1 = path0 + 'normal/' + data
   targetpath0=path1+'train/low/'+data
   targetpath1 = path1 + 'train/high/' + data

   copyFiles(sourcepath0,targetpath0)
   copyFiles(sourcepath1,targetpath1)

for data in tqdm(test):
   sourcepath0 = path0 + 'low/' + data
   sourcepath1 = path0 + 'normal/' + data
   targetpath0 = path1 + 'eval/low/' + data
   targetpath1 = path1 + 'eval/high/' + data

   copyFiles(sourcepath0, targetpath0)
   copyFiles(sourcepath1, targetpath1)