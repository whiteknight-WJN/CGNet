from raw_rgb import *
import cv2
import os
from tqdm import tqdm
import rawpy
from tqdm import tqdm

path= '../DRVtest/short'
path1 = '../DRVtest/long'

pathfile=os.listdir(path)
raw_list=[]
for path_list in pathfile:
    if path_list.endswith('.ARW'):
        raw_list.append(path+'/'+path_list)

pathfile1=os.listdir(path1)
raw_list1=[]
for path_list1 in pathfile1:
    if path_list1.endswith('.ARW'):
        raw_list1.append(path1+'/'+path_list1)

rate_list = [30,50,100]  # 合成亮度比例，取固定值
output_dir = r'../dataset_hecheng'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#
k=0
for k,raw_path1 in tqdm(enumerate(raw_list1)):
    raw_path = raw_path1.split('_')[0].replace('long','short')+'.ARW'
    if not os.path.exists(output_dir+'/'+str(k)):
        os.makedirs(output_dir+'/'+str(k))
    img_raw = rawpy.imread(raw_path)  # 读取normal RAW
    img_visible_4c = pack_raw_bayer(img_raw)
    img_raw1 = rawpy.imread(raw_path1)  # 读取normal RAW
    img_visible_4c1 = pack_raw_bayer(img_raw1)
    rate=int(np.mean(img_visible_4c1)/np.mean(img_visible_4c))

    gained_img = np.clip((img_visible_4c * rate), 0, 1)

    img3c = postprocess_bayer(raw_path, gained_img, 'srgb')
    img3c = cv2.cvtColor(img3c, cv2.COLOR_RGB2BGR)
    img3c=cv2.resize(img3c,(1080,7

                            20))
    savepath = output_dir + '/' + str(k) + '/' + str(rate) + '.png'
    cv2.imencode('.png', img3c)[1].tofile(savepath)
    gt_img = img_raw1.postprocess(use_camera_wb=True, half_size=False, no_auto_bright = True, output_bps = 8,
                                       bright = 1, user_black = None, user_sat = None)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
    gt_img = cv2.resize(gt_img, (1080, 720))
    savepath1 = output_dir + '/' + str(k) + '/' + 'origin.png'
    cv2.imencode('.png', gt_img)[1].tofile(savepath1)


#
