#!/usr/bin/python3

import  numpy as np
import os
from PIL import Image, ImageFile
import  sys
import  random
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


__all__=["load_data_detection"]
MAX_BOXES = 90

def load_data_detection(imagePath,labelPath,class_list,shape,crop,jitter,hue,saturation,exposure):
    '''
    :param imagePath:图片路径
    :param labelPath:xml label 路径
    :param class_list:类别列表
    :param shape:网络输入图片的宽高(width,height)
    :param crop:是否进行随机裁剪图片 True False
    :param jitter:抖动系数,利用抖动系数增加图片 crop为true的时候 随机剪去 shape * jitter,false 的时候随机 移动jitter * shape
    :param hue:图片旋转角度
    :param saturation: 饱和度
    :param exposure:曝光度
    :return:返回 处理过的图片 img(PIL.Image类型) label 一个numpy矩阵(maxboxes(90) * 5 矩阵,) x,y,w,h,class_index 均除以shape
    '''
    img = Image.open(imagePath).convert('RGB')
    if crop:
        ##裁剪一张图片
        img,flip,dx,dy,sx,sy = data_augmentation_crop(img,shape,jitter,hue,saturation,exposure)
    else:
        img, flip, dx, dy, sx, sy = data_augmentation_nocrop(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labelPath, class_list,crop, flip, -dx, -dy, sx, sy)
    return img,label



def fill_truth_detection(labelPath,classes,crop, flip, dx, dy, sx, sy,keep_difficult = True):
    max_boxes = MAX_BOXES
    label_map = np.zeros((max_boxes, 5))

    target_root = ET.parse(labelPath).getroot()
    size_root = target_root.find('size')
    width = int(size_root.find('width').text)
    height = int(size_root.find('height').text)

    box_count = 0
    for obj in target_root.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if difficult and (not keep_difficult):
            continue
        name = obj[0].text.strip()
        name_index = classes.index(name)
        bbox = obj[4]

        # bndbox = [int(bb.text)  for bb in bbox]
        xmin = int(bbox.find('xmin').text) / width
        x1 = min(0.999, max(0, xmin * sx - dx))
        xmax = int(bbox.find('xmax').text) / width
        x2 = min(0.999, max(0, xmax * sx - dx))
        ymin = int(bbox.find('ymin').text) / height
        y1 = min(0.999, max(0, ymin * sy - dy))
        ymax = int(bbox.find('ymax').text) / height
        y2 = min(0.999, max(0, ymax * sy - dy))

        x = (x1 + x2) / 2  # center x
        y = (y1 + y2) / 2  # center y
        w = (x2 - x1)  # width
        h = (y2 - y1)  # height
        if flip:
            x = 0.999 - x
            # when crop is applied, we should check the cropped width/height ratio
        if w < 0.0001 or h < 0.0001 or \
                (crop and (x / h > 20 or h / w > 20)):
            continue
        label_map[box_count][0] = x
        label_map[box_count][1] = y
        label_map[box_count][2] = w
        label_map[box_count][3] = h
        label_map[box_count][4] = name_index
        box_count += 1
    return label_map
def data_augmentation_crop(img,shape,jitter,hue,saturation,exposure):

    old_w = img.width
    old_h = img.height

    dw = int(old_w * jitter)
    dh = int(old_h * jitter)

    pleft = random.randint(-dw,dw)
    pright = random.randint(-dw,dw)
    ptop = random.randint(-dh,dh)
    pbottom = random.randint(-dh,dh)

    rest_width = old_w - pleft - pright
    rest_height = old_h - ptop - pbottom

    sx = old_w / rest_width
    sy = old_h / rest_height

    flip = random.randint(0,1)

    ##裁剪图片box
    cropbb = np.array([pleft, ptop, pleft + rest_width - 1, ptop + rest_height - 1])

    new_h,new_w = cropbb[2]-cropbb[0],cropbb[3]-cropbb[1]

    cropbb[0] = -min(cropbb[0],0)
    cropbb[1] = -min(cropbb[1],0)
    cropbb[2] = min(cropbb[2],old_w)
    cropbb[3] = min(cropbb[3],old_h)

    cropped = img.crop(cropbb)

    left_top_point =(pleft if pleft >0 else 0 ,ptop if ptop > 0 else 0)

    new_img = Image.new("RGB",(new_w,new_h),(127,127,127))
    new_img.paste(cropped,left_top_point)
    sized = new_img.resize(shape)

    del  cropped,new_img

    ##new left point 在新图片的比例
    dx = (float(pleft) / old_w ) * sx
    dy =(float(ptop) / old_h ) * sy

    if flip:
        ##左右翻转
        sized =sized.transpose(Image.FLIP_LEFT_RIGHT)

    ##调整角度饱和度曝光率等
    img = random_distort_image(sized, hue, saturation, exposure)
    return  img,flip,dx,dy,sx,sy


def data_augmentation_nocrop(img, shape, jitter, hue, sat, exp):
    net_w, net_h = shape
    img_w, img_h = img.width, img.height

    # determine the amount of scaling and cropping
    dw = jitter * img_w
    dh = jitter * img_h

    ##原来图片高小于宽的话,调整后图片宽保持net_w,高按比例收缩
    new_ar = (img_w + random.uniform(-dw, dw)) / (img_h + random.uniform(-dh, dh))
    # scale = np.random.uniform(0.25, 2)
    scale = 1.

    ##调整新区域
    if (new_ar < 1):
        new_h = int(scale * net_h)
        new_w = int(net_h * new_ar)
    else:
        new_w = int(scale * net_w)
        new_h = int(net_w / new_ar)

    ##填充部分
    dx = int(random.uniform(0, net_w - new_w))
    dy = int(random.uniform(0, net_h - new_h))
    sx, sy = new_w / net_w, new_h / net_h

    # 变形已经随机移动图片整体部分到canvas中
    new_img = image_scale_and_shift(img, new_w, new_h, net_w, net_h, dx, dy)

    # randomly distort hsv space
    new_img = random_distort_image(new_img, hue, sat, exp)

    # randomly flip
    flip = random.randint(0, 1)
    if flip:
        ##是否水平翻转图像
        new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)

    dx, dy = dx / net_w, dy / net_h
    return new_img, flip, dx, dy, sx, sy






##读取label 信息
def fill_truth_detection(labelPath,classes,crop, flip, dx, dy, sx, sy,keep_difficult = True):
    label_map = np.zeros((MAX_BOXES, 5))

    target_root = ET.parse(labelPath).getroot()
    size_root = target_root.find('size')
    width = int(size_root.find('width').text)
    height = int(size_root.find('height').text)

    box_count = 0
    for obj in target_root.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if difficult and (not keep_difficult):
            continue
        name = obj[0].text.strip()
        name_index = classes.index(name)
        bbox = obj[4]

        # bndbox = [int(bb.text)  for bb in bbox]
        xmin = int(bbox.find('xmin').text) / width
        x1 = min(0.999, max(0, xmin * sx - dx))
        xmax = int(bbox.find('xmax').text) / width
        x2 = min(0.999, max(0, xmax * sx - dx))
        ymin = int(bbox.find('ymin').text) / height
        y1 = min(0.999, max(0, ymin * sy - dy))
        ymax = int(bbox.find('ymax').text) / height
        y2 = min(0.999, max(0, ymax * sy - dy))

        x = (x1 + x2) / 2  # center x
        y = (y1 + y2) / 2  # center y
        w = (x2 - x1)  # width
        h = (y2 - y1)  # height
        if flip:
            x = 0.999 - x
            # when crop is applied, we should check the cropped width/height ratio
        if w < 0.0001 or h < 0.0001 or \
                (crop and (x / h > 20 or h / w > 20)):
            continue
        label_map[box_count][0] = x
        label_map[box_count][1] = y
        label_map[box_count][2] = w
        label_map[box_count][3] = h
        label_map[box_count][4] = name_index
        box_count += 1
        if box_count > MAX_BOXES:
            break
    return label_map





##调整图片系列

def image_scale_and_shift(img, new_w, new_h, net_w, net_h, dx, dy):
    scaled = img.resize((new_w, new_h))
    # find to be cropped area
    sx, sy = -dx if dx < 0 else 0, -dy if dy < 0 else 0
    ex, ey = new_w if sx+new_w<=net_w else net_w-sx, new_h if sy+new_h<=net_h else net_h-sy
    scaled = scaled.crop((sx, sy, ex, ey))

    # find the paste position
    sx, sy = dx if dx > 0 else 0, dy if dy > 0 else 0
    assert sx+scaled.width<=net_w and sy+scaled.height<=net_h
    new_img = Image.new("RGB", (net_w, net_h), (127, 127, 127))
    new_img.paste(scaled, (sx, sy))
    del scaled
    return new_img

def random_distort_image(im, hue, saturation, exposure):
    dhue = np.random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def rand_scale(s):
    scale = np.random.uniform(1, s)
    if np.random.randint(2):
        return scale
    return 1./scale


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    # constrain_image(im)
    return im

