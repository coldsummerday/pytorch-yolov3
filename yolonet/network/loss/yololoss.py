#!/usr/bin/python3
import numpy as np

import torch
import torch.nn as nn
from ...data.boxes import Detection

import time
import logging as log

import math
class YoloLayer(nn.Module):
    def __init__(self, anchors, hyperparams,scale = 8):
        """

        :param anchors:本层使用的anchor,以[(x,y),(x,y)]的形式
        :param hyperparams:超参数类
        """
        super(YoloLayer, self).__init__()
        self.device = torch.device("cuda" if hyperparams.yolo_layer_cuda else "cpu")
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = hyperparams.classes
        """
        :arg
        """
        self.ignore_thresh = hyperparams.ignore_thresh
        self.truth_thresh = hyperparams.truth_thresh

        self.rescore = hyperparams.rescore

        self.max_boxes = hyperparams.max_boxes
        self.conf_thresh = hyperparams.conf_thresh

        self.layer_width = 0
        self.layer_height = 0
        self.net_width = hyperparams.net_width
        self.net_height = hyperparams.net_height

        self.yolo_loss = Yolo_loss()

        self.scale = scale

        self._stride_layer_dcit = {
            32: "82",
            16: "94",
            8: '106'
        }

    def forward(self, x, target=None):
        if isinstance(target,torch.Tensor):
            return self._train_forward(x, target)
        else:
            return self._detect_forward(x)

    def _detect_forward(self, x):

        start_time = time.time()
        batch_size = x.data.size(0)
        detectBoxs = []
        self.layer_width, self.layer_height = x.data.size(3), x.data.size(2)

        cls_anchor_dim = batch_size * self.num_anchors * self.layer_height * self.layer_width
        output = x.view(batch_size * self.num_anchors, (4 + 1 + self.num_classes), self.layer_height * self.layer_width
                        ).transpose(0, 1).contiguous().view(4 + 1 + self.num_classes, cls_anchor_dim).to(self.device)

        grid_x = torch.linspace(0, self.layer_width - 1, self.layer_width).repeat(batch_size * self.num_anchors,
                                                                                  self.layer_height, 1) \
            .view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, self.layer_height - 1, self.layer_height).repeat(self.layer_width, 1) \
            .t().repeat(batch_size * self.num_anchors, 1, 1).view(cls_anchor_dim).to(self.device)

        anchor_w = torch.FloatTensor(
            [[anchor[0] for i in range(0, batch_size * self.layer_width * self.layer_height)] for anchor in
             self.anchors]) \
            .view(cls_anchor_dim).to(self.device)
        anchor_h = torch.FloatTensor(
            [[anchor[1] for i in range(0, batch_size * self.layer_width * self.layer_height)] for anchor in
             self.anchors]) \
            .view(cls_anchor_dim).to(self.device)

        x_vec = (output[0].sigmoid() + grid_x) / self.layer_width
        y_vec = (output[1].sigmoid() + grid_y) / self.layer_height
        w_vec = (output[2].exp() * anchor_w) / self.net_width
        h_vec = (output[3].exp() * anchor_h) / self.net_height
        objectness_list = output[4].sigmoid()
        cls_confs = torch.nn.Softmax(dim=1)(output[5:5 + self.num_classes].transpose(0, 1)).detach()

        bacth_dim = self.num_anchors * self.layer_width * self.layer_height
        for bacth_index in range(batch_size):
            #应该针对每个batch 做单独判断
            batch_detections = []
            for cy in range(self.layer_height):
                for cx in range(self.layer_width):
                    for anchor_i in range(self.num_anchors):
                        # 第几个predboxes
                        index = bacth_index * bacth_dim \
                                + anchor_i * self.layer_width * self.layer_height \
                                + cy * self.layer_width + cx
                        objectness = objectness_list[index]
                        if objectness <= self.conf_thresh:
                            continue
                        box_c_x = float(x_vec[index])
                        box_c_y = float(y_vec[index])
                        box_c_w = float(w_vec[index])
                        box_c_h = float(h_vec[index])
                        if (box_c_x > 1 or box_c_y > 1 or box_c_w > 1 or box_c_h > 1):
                            continue

                        line = [box_c_x,box_c_y,box_c_w,box_c_h,float(objectness)]
                        # 每个类的得分的判断
                        prob_list = []
                        for i in range(self.num_classes):
                            prob = float(objectness * cls_confs[index][i])
                            prob = float(prob) if prob > self.conf_thresh else 0
                            prob_list.append(prob)
                        line.extend(prob_list)
                        batch_detections.append(line)
            detectBoxs.append(batch_detections)
        spend_time = time.time() - start_time
        log.warn("{} layer spend time:{}s".format(self._stride_layer_dcit.get(self.net_height / self.layer_height),
                                                  spend_time))
        # print("{} layer spend time:{}s".format(self._stride_layer_dcit.get(self.net_height / self.layer_height),spend_time))
        return detectBoxs

    def _train_forward(self, x, target=None):
        # output : batch_size * B*(4+1+num_classes)*H*W
        # B =3每层选择3个anchor
        # 获取loss
        # start_time = time.time()
        nRecall = 0
        nRecall75 = 0
        num_gt = 0
        avg_iou = 0.0

        batch_size = x.data.size(0)
        #log.warn(x.data.size())
        self.layer_width, self.layer_height = x.data.size(3), x.data.size(2)
        self.net_height,self.net_width = self.scale * self.layer_height,self.scale*self.layer_width

        num_image_pixel = self.num_anchors * self.layer_width * self.layer_height


        no_obj_mask = torch.ones(batch_size, self.num_anchors, self.layer_height * self.layer_width)
        obj_mask = torch.zeros(batch_size, self.num_anchors, self.layer_height * self.layer_width)
        coord_mask = torch.zeros(batch_size, self.num_anchors, self.layer_height * self.layer_width)

        truth_coord = torch.zeros(4, batch_size, self.num_anchors, self.layer_height * self.layer_width)
        truth_conf = torch.zeros(batch_size, self.num_anchors, self.layer_height * self.layer_width)
        truth_cls = torch.zeros(batch_size, self.num_anchors, self.layer_height * self.layer_width)

        cls_anchor_dim = batch_size * self.num_anchors * self.layer_height * self.layer_width
        output = x.view(batch_size, self.num_anchors, (4 + 1 + self.num_classes), self.layer_height,
                        self.layer_width).to(self.device)
        '''
        cls_grid tensor([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14], device='cuda:0')
        '''

        cls_grid = torch.linspace(5, 5 + self.num_classes - 1, self.num_classes).long().to(self.device)
        pred_boxes_id = torch.LongTensor(range(0, 5)).to(self.device)
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim).to(self.device)

        # yolo_boxes 处理
        coord = output.index_select(2, pred_boxes_id[0:4]) \
            .view(batch_size * self.num_anchors, -1, self.layer_height * self.layer_width) \
            .transpose(0, 1).contiguous().view(-1, cls_anchor_dim).to(self.device)

        coord[0:2] = coord[0:2].sigmoid()
        confidences = output.index_select(2, pred_boxes_id[4]).view(cls_anchor_dim).sigmoid()

        class_vec = output.index_select(2, cls_grid)
        class_vec = class_vec.view(batch_size * self.num_anchors, self.num_classes,
                                   self.layer_width * self.layer_height) \
            .transpose(1, 2).contiguous().view(cls_anchor_dim, self.num_classes)

        class_vec = torch.nn.Softmax(dim=1)(class_vec)

        grid_x = torch.linspace(0, self.layer_width - 1, self.layer_width) \
            .repeat(batch_size * self.num_anchors, self.layer_height, 1).view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, self.layer_height - 1, self.layer_height) \
            .repeat(self.layer_width, 1).t().repeat(batch_size * self.num_anchors, 1).view(cls_anchor_dim).to(
            self.device)

        anchor_w = torch.FloatTensor(
            [[anchor[0] for i in range(0, batch_size * self.layer_width * self.layer_height)] for anchor in
             self.anchors]) \
            .view(cls_anchor_dim).to(self.device)
        anchor_h = torch.FloatTensor(
            [[anchor[1] for i in range(0, batch_size * self.layer_width * self.layer_height)] for anchor in
             self.anchors]) \
            .view(cls_anchor_dim).to(self.device)

        '''
        yolo 源码中:    
        b.x = (i + x[index + 0*stride]) / lw;
        b.y = (j + x[index + 1*stride]) / lh;
        b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
        b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;

        '''
        pred_boxes[0] = (coord[0] + grid_x) / self.layer_width
        pred_boxes[1] = (coord[1] + grid_y) / self.layer_height
        pred_boxes[2] = (coord[2].exp() * anchor_w) / self.net_width
        pred_boxes[3] = (coord[3].exp() * anchor_h) / self.net_height

        ## 此处应该抽象一个get yolo box 方法

        del grid_x, grid_y, anchor_h, anchor_w
        pred_boxes = pred_boxes.transpose(0, 1).contiguous().view(-1, 4).to(
            self.device).detach()
        ##ground_truth 用框占比例来算
        for batch_index in range(batch_size):
            current_pred_boxes = pred_boxes[batch_index * num_image_pixel:(batch_index + 1) * num_image_pixel]
            current_ious = torch.zeros(num_image_pixel)
            gruth_truth_boxes = target[batch_index].view(-1, 5).to("cpu")

            for t in range(self.max_boxes):
                if gruth_truth_boxes[t][0] == 0:
                    break
                groud_truth_x = gruth_truth_boxes[t][0]
                groud_truth_y = gruth_truth_boxes[t][1]
                groud_truth_w = gruth_truth_boxes[t][2]
                groud_truth_h = gruth_truth_boxes[t][3]

                current_gt_boxes = torch.FloatTensor([groud_truth_x, groud_truth_y, groud_truth_w, groud_truth_h]) \
                    .repeat(num_image_pixel, 1)

                ##为了找 与该真实框iou 最大的预测框
                current_ious = torch.max(current_ious,
                                         bbox_iou_mutil_center(current_pred_boxes, current_gt_boxes))

            ignore_index = (current_ious > self.ignore_thresh).view(num_image_pixel)

            ##iou > ignore_thresh ,不参加loss计算
            no_obj_mask[batch_index][ignore_index.view(self.num_anchors, -1)] = 0

            for t in range(self.max_boxes):

                if gruth_truth_boxes[t][0] == 0:
                    break
                num_gt += 1

                groud_truth_x = gruth_truth_boxes[t][0]
                groud_truth_y = gruth_truth_boxes[t][1]
                groud_truth_w = gruth_truth_boxes[t][2]
                groud_truth_h = gruth_truth_boxes[t][3]

                groud_truth_w, groud_truth_h = groud_truth_w.float(), groud_truth_h.float()
                gi, gj = int(groud_truth_x * self.layer_width), int(groud_truth_y * self.layer_height)

                ##找到最合适的anchor
                tmp_gt_boxes = torch.FloatTensor([0, 0, groud_truth_w, groud_truth_h]).repeat(self.num_anchors, 1)
                anchors_list = [[0, 0, anchor[0] / self.net_width, anchor[1] / self.net_height] for anchor in
                                self.anchors]
                tmp_anchors_boxes = torch.FloatTensor(anchors_list)
                _, best_index = torch.max(bbox_iou_mutil_center(tmp_anchors_boxes, tmp_gt_boxes), 0)
                best_anchor_box = anchors_list[int(best_index)]
                best_anchor_w = best_anchor_box[2] * self.net_width
                best_anchor_h = best_anchor_box[3] * self.net_height
                # 第几个anchor比较好(0,1,2)

                groud_truth_box = torch.FloatTensor([groud_truth_x, groud_truth_y, groud_truth_w, groud_truth_h])
                pred_box = pred_boxes[batch_index * self.num_anchors * self.layer_width * self.layer_height +
                                      best_index * self.layer_height * self.layer_width + gj * self.layer_width + gi]

                # 计算单个iou
                iou = bbox_iou_center(groud_truth_box, pred_box)

                avg_iou += iou
                obj_mask[batch_index][best_index][gj * self.layer_width + gi] = 1
                no_obj_mask[batch_index][best_index][gj * self.layer_width + gi] = 0

                coord_mask[batch_index][best_index][gj * self.layer_width + gi] \
                    = 2.0 - groud_truth_w * groud_truth_h

                # ground_truth 在yolov3 中的计算方式
                '''
                 //分别计算真实物体的 xywh
            float tx = (truth.x*lw - i);
            float ty = (truth.y*lh - j);
            float tw = log(truth.w*w / biases[2*n]);
            float th = log(truth.h*h / biases[2*n + 1]);
                '''

                # truth_coord
                truth_coord[0][batch_index][best_index][gj * self.layer_width + gi] \
                    = groud_truth_x * self.layer_width - gi
                truth_coord[1][batch_index][best_index][gj * self.layer_width + gi] \
                    = groud_truth_y * self.layer_width - gj
                truth_coord[2][batch_index][best_index][gj * self.layer_width + gi] \
                    = math.log(groud_truth_w * self.net_width / best_anchor_w)
                truth_coord[3][batch_index][best_index][gj * self.layer_width + gi] \
                    = math.log(groud_truth_h * self.net_height / best_anchor_h)

                # truth_class
                truth_cls[batch_index][best_index][gj * self.layer_width + gi] \
                    = gruth_truth_boxes[t][4]
                # confidence
                truth_conf[batch_index][best_index][gj * self.layer_width + gi] \
                    = iou if self.rescore else 1
                if iou > 0.5:
                    nRecall += 1
                    if iou > 0.75:
                        nRecall75 += 1


        class_mask = (obj_mask == 1)
        truth_cls_index = truth_cls[class_mask].long().view(-1).to(self.device)

        # 每个预测框的class [0-1]
        class_mask = class_mask.view(-1, 1).repeat(1, self.num_classes).to(self.device)
        class_vec = class_vec[class_mask].view(-1, self.num_classes)

        truth_cls_vec = torch.zeros(len(truth_cls_index), self.num_classes).to(self.device)
        truth_cls_vec[range(len(truth_cls_index)), truth_cls_index] = 1

        truth_coord = truth_coord.view(4, cls_anchor_dim).to(self.device)

        truth_conf = truth_conf.view(cls_anchor_dim).to(self.device)

        conf_mask = (obj_mask + no_obj_mask).view(cls_anchor_dim).to(self.device)
        coord_mask = coord_mask.view(cls_anchor_dim).to(self.device)

        yolo_total_loss = self.yolo_loss(coord * coord_mask, truth_coord * coord_mask,
                                         confidences * conf_mask, truth_conf * conf_mask,
                                         class_vec, truth_cls_vec, batch_size)
        '''
        self.cache['coordloss'] = loss_coord.sum().item() / batch_size
        self.cache["confloss"] = loss_conf.sum().item() / batch_size
        self.cache["labelloss"] = loss_class.sum().item() / batch_size
        '''

        #log.warn(loss_cache)
        #log.warn((self.net_height,self.layer_height))


        if num_gt==0:
            num_gt=1

        layer_index = self._stride_layer_dcit.get(self.scale)
        info_str = "Region {} Avg IOU: {:.3}\t" \
                   "recall.5R: {:.3},\trecall.75R: {:.3} count: {} "\
            .format(layer_index,avg_iou/num_gt,nRecall/num_gt,
                    nRecall75 / num_gt,num_gt
                )
        log.info(info_str)

        #print("Region %s Avg IOU: %f  .5R: %f, .75R: %f,  count: %d" % (
        #    layer_index, avg_iou / num_gt, nRecall / num_gt, nRecall75 / num_gt, num_gt
        #))
        # spend_time = time.time()-start_time
        # print("{} layer spend time:{} s".format(layer_index,spend_time))
        return yolo_total_loss


class Yolo_loss(nn.Module):
    def __init__(self,coord_scale = 1,class_scale = 1,conf_scale = 1):
        super(Yolo_loss,self).__init__()

        self.cache = {}


        ##w,h出现loss
        self.coord_scale = coord_scale
        self.class_scale = class_scale
        self.conf_scale = conf_scale

        self.class_loss_f = nn.MSELoss(reduction="sum")
        self.conf_class_f = nn.MSELoss(reduction="sum")
        #用多分类的交叉熵函数loss
        self.loss_coord_x_f =nn.MSELoss(reduction='sum')
        self.loss_coord_y_f = nn.MSELoss(reduction='sum')
        self.loss_coord_w_f = nn.SmoothL1Loss(reduction='sum')
        self.loss_coord_h_f = nn.SmoothL1Loss(reduction='sum')


    def forward(self,coord,truth_coord,confidences,truth_conf,class_vec,truth_cls,batch_size):

        loss_x = self.loss_coord_x_f(coord[0],truth_coord[0]) / batch_size

        loss_y = self.loss_coord_y_f(coord[1],truth_coord[1]) / batch_size


        coord_w = coord[2]
        truth_w = truth_coord[2]
        # 先取绝对值再进行开平方操作,这样就不会有nan了, 然后恢复各项的符号   ......会有nan

        #coord_w = coord_w.abs().sqrt() * get_positive_and_negative_tensor(coord_w)
        #truth_w = truth_w.abs().sqrt() * get_positive_and_negative_tensor(truth_w)
        loss_w = self.loss_coord_w_f(coord_w, truth_w) / batch_size

        coord_h = coord[3]
        truth_h = truth_coord[3]


        #coord_h = coord_h.abs().sqrt() * get_positive_and_negative_tensor(coord_h)
        #truth_h = truth_h.abs().sqrt() * get_positive_and_negative_tensor(truth_h)
        loss_h = self.loss_coord_h_f(coord_h, truth_h) / batch_size

        loss_coord = self.coord_scale * (loss_x + loss_y + loss_w + loss_h)



        loss_conf = self.conf_scale *  self.conf_class_f(confidences ,truth_conf ) /batch_size

        loss_class = self.class_scale * self.class_loss_f(class_vec,truth_cls) / batch_size

        total_loss = loss_coord + loss_conf + loss_class

        return total_loss


'''
//delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);
void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        //正确的class_index,loss = 0
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){
        //class 错了,loss 就等于预测的结果
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}
'''


def get_positive_and_negative_tensor(input_tensor):
    a = torch.ones(input_tensor.size()).type_as(input_tensor)
    b = ((input_tensor  < 0) * 2).type_as(input_tensor)
    return  a-b

def bbox_iou_mutil_center(boxA,boxB):
    #获取corner 信息
    a_x1 = boxA[:,0] - boxA[:,2] / 2.0
    a_y1 = boxA[:,1] - boxA[:,3] / 2.0
    a_x2 = boxA[:,0] + boxA[:,2] / 2.0
    a_y2 = boxA[:,1] + boxA[:,3] / 2.0


    b_x1 = boxB[:,0] - boxB[:,2] / 2.0
    b_y1 = boxB[:,1] - boxB[:,3] / 2.0
    b_x2 = boxB[:,0] + boxB[:,2] / 2.0
    b_y2 = boxB[:,1] + boxB[:,3] / 2.0

    #相交部分
    inter_rect_x1 =  torch.max(a_x1, b_x1)
    inter_rect_y1 =  torch.max(a_y1, b_y1)
    inter_rect_x2 =  torch.min(a_x2, b_x2)
    inter_rect_y2 =  torch.min(a_y2, b_y2)

    #相交面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 ,min=0) *\
                torch.clamp(inter_rect_y2 - inter_rect_y1 ,min=0)

    #各自原来面积
    boxa_area = (a_x2 - a_x1) * (a_y2 - a_y1 )
    boxb_area = (b_x2 - b_x1 ) * (b_y2 - b_y1 )
    iou = inter_area / (boxa_area + boxb_area - inter_area + 0.000001)
    return iou


def bbox_iou_center(boxA,boxB):
    a_x1 = boxA[0] - boxA[2] / 2.0
    a_y1 = boxA[1] - boxA[3] / 2.0
    a_x2 = boxA[0] + boxA[2] / 2.0
    a_y2 = boxA[1] + boxA[3] / 2.0

    b_x1 = boxB[0] - boxB[2] / 2.0
    b_y1 = boxB[1] - boxB[3] / 2.0
    b_x2 = boxB[0] + boxB[2] / 2.0
    b_y2 = boxB[1] + boxB[3] / 2.0

    # 相交部分
    inter_rect_x1 = torch.max(a_x1, b_x1)
    inter_rect_y1 = torch.max(a_y1, b_y1)
    inter_rect_x2 = torch.min(a_x2, b_x2)
    inter_rect_y2 = torch.min(a_y2, b_y2)

    # 相交面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # 各自原来面积
    boxa_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    boxb_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    iou = inter_area / (boxa_area + boxb_area - inter_area + 0.000001)
    return iou

def clamp(x,min=0,max=1):
    if x <= min:
        return 0
    elif  x > min and x < max:
        return x
    elif x>=max:
        return max







