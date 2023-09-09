import os

import numpy as np
import rawpy
import cv2


class Logger():
    def __init__(self, f_handle):
        super(Logger, self).__init__()
        self.f_handle = f_handle

    def write(self, str):
        print(str)
        print(str, file=self.f_handle)


def bayer_to_img4c(bayer):
    img4c = np.ndarray((bayer.shape[0] // 2, bayer.shape[1] // 2, 4), dtype
=bayer.dtype)
    img4c[:, :, 0] = bayer[0::2, 0::2, ...]
    img4c[:, :, 1] = bayer[0::2, 1::2, ...]
    img4c[:, :, 2] = bayer[1::2, 0::2, ...]
    img4c[:, :, 3] = bayer[1::2, 1::2, ...]
    return img4c


def pack_raw_bayer(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)

    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2, R[1][0]:W:2],  # RGBG
                    im[G1[0][0]:H:2, G1[1][0]:W:2],
                    im[B[0][0]:H:2, B[1][0]:W:2],
                    im[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0).astype(np.float32
                                                                    )

    white_point = raw.white_level
    black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)# print(black_level[0,0,0], white_point)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)

    return out


def postprocess_bayer(rawpath, img4c, flag='rawrgb'):
    img4c = np.clip(img4c, 0, 1)

    # unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    G2 = np.where(raw_pattern == 3)
    B = np.where(raw_pattern == 2)
    #
    white_point = raw.white_level
    black_level = np.array(raw.black_level_per_channel)[:, None, None]
    img4c = np.minimum(img4c * (white_point - black_level) + black_level,white_point)
    img4c = img4c.astype(np.uint16)

    img_shape = raw.raw_image_visible.shape
    H = img_shape[0]
    W = img_shape[1]

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :, :]
    raw.raw_image_visible[G1[0][0]:H:2, G1[1][0]:W:2] = img4c[1, :, :]
    raw.raw_image_visible[B[0][0]:H:2, B[1][0]:W:2] = img4c[2, :, :]
    raw.raw_image_visible[G2[0][0]:H:2, G2[1][0]:W:2] = img4c[3, :, :]

    if flag == 'rawrgb':
        out = np.concatenate([img4c[0:1], (0.5 * (img4c[1:2] + img4c[3:])),
                              img4c[2:3]], 0).transpose(1, 2, 0)*255
        out = np.uint8(out)
    else:
        out = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright = True, output_bps = 8,
                                       bright = 1, user_black = None, user_sat = None)

    return out

