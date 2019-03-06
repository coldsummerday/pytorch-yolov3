

import random
import logging as log
import numpy as np
import torch
from  PIL  import  Image,ImageOps

try:
    import cv2
except ImportError:
    log.warn('OpenCV is not installed and cannot be used')
    cv2 = None

class LetterBoxImage(object):
    def __init__(self,net_size):
        self.net_size = net_size
        self.fill_color = 127


    def __call__(self,data):
        if data is None:
            return None
        elif isinstance(data,Image.Image):
            return self._transfrom_pil(data)
        elif isinstance(data,np.ndarray):
            return self._transform_cv(data)

    def _transform_cv(self,img):
        img_w,img_h = img.shape[1],img.shape[0]
        net_h,net_w = self.net_size
        scale = min(net_w/img_w,net_h/img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if new_h==net_h and new_w == net_w:
            return resized_image

        #padding

        canvas = np.full((net_h, net_w, 3), self.fill_color)

        h_start_index = (net_h - new_h) // 2
        h_end_index = (net_h - new_h) // 2 + new_h

        w_start_index = (net_w - new_w)//2
        w_end_index = (net_w - new_w) // 2 + new_w

        canvas[h_start_index:h_end_index,w_start_index:w_end_index,:] = resized_image

        return canvas

    def _transfrom_pil(self,img):
        img_w,img_h = img.size
        net_w,net_h = self.net_size

        scale = min(net_w/img_w,net_h/img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resample_mode = Image.NEAREST #Image.BILINEAR if self.scale > 1 else Image.ANTIALIAS
        resize_img = img.resize((new_w, new_h), resample_mode)

        if new_h==net_h and new_w == net_w:
            return resize_img

        #padding

        channels = len(resize_img.getbands())
        pad_w = (net_w - new_w) / 2
        pad_h = (net_h - new_h) / 2
        pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        canvas = ImageOps.expand(resize_img, border=pad, fill=(self.fill_color,) * channels)
        return canvas











