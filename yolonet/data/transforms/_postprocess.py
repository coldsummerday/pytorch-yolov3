

from ..boxes.detection import Detection
import torch
from PIL import Image,ImageDraw
import math
import  time
import  logging as log
__all__=["CorrectDetection","get_all_pred_boxes","DrawRectengle"]

class CorrectDetection:
    def __init__(self,net_size):

        self.net_size = net_size

    def __call__(self,detections,img_size):
        """
        nms 后的每条记录为[x,y,w,h,confidence,classid]
        :param detections:包含每条记录的list
        :param img_size: 包含原来图像的元组(img_w,img_h)
        :return:
        """
        img_width, img_height = img_size

        net_height,net_width = self.net_size

        if net_width / img_width < net_height / img_height:
            new_width = net_width
            new_height = (img_height * net_width) / img_width
        else:
            new_width = (img_width * net_height) / img_height
            new_height = net_height
        x_old, x_slide = (new_width - new_width) / (2 * net_width), net_width / new_width
        y_old, y_slide = (net_height - new_height) / (2 * net_height), net_height / new_height
        for i in range(len(detections)):
            if detections[i][4] == 0:
                continue
            detections[i][0] = (detections[i][0] - x_old) * x_slide
            detections[i][1] = (detections[i][1] - y_old) * y_slide
            detections[i][2] *= x_slide
            detections[i][3] *= y_slide
        return detections

def get_all_pred_boxes(output,anchors,net_size,num_classes,conf_thresh=0.5):
    '''
    :param outputs: model output features
    :param anchors: 本层使用的anchor,以[(w,h),(w,h)]的形式
    :param net_size:网络input_shape (w,h)
    :param num_classes:int ,
    :param conf_thresh
    :return:
    '''
    start_time = time.time()
    num_anchors = len(anchors)
    net_width,net_height = net_size
    batch_size = output.data.size(0)
    layer_width,layer_height = output.data.size(3),output.data.size(2)

    #使用cpu 会快于GPU
    device = torch.device('cpu')


    cls_anchor_dim = batch_size * num_anchors * layer_height * layer_width

    output = output.view(batch_size * num_anchors, (4 + 1 + num_classes), layer_height * layer_width
                    ).transpose(0, 1).contiguous().view(4 + 1 + num_classes, cls_anchor_dim).to(device)

    grid_x = torch.linspace(0, layer_width - 1, layer_width).repeat(batch_size * num_anchors,
                                                                              layer_height, 1) \
        .view(cls_anchor_dim).to(device)
    grid_y = torch.linspace(0, layer_height - 1, layer_height).repeat(layer_width, 1) \
        .t().repeat(batch_size * num_anchors, 1, 1).view(cls_anchor_dim).to(device)

    anchor_w = torch.FloatTensor(
        [[anchor[0] for i in range(0, batch_size * layer_width * layer_height)] for anchor in
         anchors]) \
        .view(cls_anchor_dim).to(device)
    anchor_h = torch.FloatTensor(
        [[anchor[1] for i in range(0, batch_size * layer_width * layer_height)] for anchor in
         anchors]) \
        .view(cls_anchor_dim).to(device)

    x_vec = (output[0].sigmoid() + grid_x) / layer_width
    y_vec = (output[1].sigmoid() + grid_y) / layer_height
    w_vec = (output[2].exp() * anchor_w) / net_width
    h_vec = (output[3].exp() * anchor_h) / net_height
    objectness_list = output[4].sigmoid()
    cls_confs = torch.nn.Softmax(dim=1)(output[5:5 + num_classes].transpose(0, 1)).detach()
    result = []
    bacth_dim = num_anchors * layer_width * layer_height
    for bacth_index in range(batch_size):

        batch_detections = []
        for cy in range(layer_height):
            for cx in range(layer_width):
                for anchor_i in range(num_anchors):
                    # 第几个predboxes
                    index = bacth_index * bacth_dim \
                            + anchor_i * layer_width * layer_height \
                            + cy * layer_width + cx
                    objectness = objectness_list[index]
                    if objectness <= conf_thresh:
                        continue
                    box_c_x = float(x_vec[index])
                    box_c_y = float(y_vec[index])
                    box_c_w = float(w_vec[index])
                    box_c_h = float(h_vec[index])
                    if (box_c_x > 1 or box_c_y > 1 or box_c_w > 1 or box_c_h > 1):
                        continue

                    line = [box_c_x, box_c_y, box_c_w, box_c_h, float(objectness)]
                    # 每个类的得分的判断
                    prob_list = []
                    for i in range(num_classes):
                        prob = float(objectness * cls_confs[index][i])
                        prob = float(prob) if prob > conf_thresh else 0
                        prob_list.append(prob)
                    line.extend(prob_list)
                    batch_detections.append(line)
        result.append(batch_detections)
    end_time = time.time()
    #log.debug("get_yolo_boxes,spend time:{}".format(end_time-start_time))
    return result




def DrawRectengle(ori_img,detections,lables):

    ori_width,ori_height = ori_img.size
    draw_handle = ImageDraw.Draw(ori_img)
    for box_line in detections:
        x, y, w, h, conf,label_index = box_line
        xmin = int( (x-w/2) * ori_width)
        ymin = int((y-h/2) * ori_height)
        xmax = int((x + w/2) *ori_width)
        ymax = int((y + h/2 ) * ori_height)
        label = lables[label_index]
        draw_handle.rectangle([(xmin,ymin),(xmax,ymax)],outline=_get_rgb_color(label_index,len(lables)),width=5)
        draw_handle.text((xmin, ymin-10), label,10)
    return ori_img


def _get_rgb_color(class_index,classes):
    offset = class_index * 123457 % classes
    red = get_color(2, offset, classes)
    green = get_color(1, offset, classes)
    blue = get_color(0, offset, classes)
    return (red,green,blue)

def get_color(c, x, max_val):
    colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    ratio = float(x)/max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
    return int(r*255)




