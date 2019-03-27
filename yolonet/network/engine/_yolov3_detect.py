#!/usr/bin/python3


from  PIL import Image

from  yolonet.network.module import  Yolov3_abc
import torch
from  yolonet.data.transforms._preprocess import LetterBoxImage
from yolonet.utils import initEnv
import torchvision.transforms as transforms
from yolonet.data.transforms._postprocess import CorrectDetection,DrawRectengle,get_all_pred_boxes
from yolonet.data.transforms.yolonms import nms
from yolonet.data.boxes.detection import DetectionResult
from yolonet.data.dataset import  VOCDetectionSet
import numpy as np

from  yolonet import  HyperParams
class YoloV3DetectEngine(object):
    def __init__(self,config):

        self.hyper_params = HyperParams(config,train_flag=2)
        self.input_shape = config["input_shape"]
        self.anchors = self.hyper_params.anchors
        self.anchor_mask = self.hyper_params.anchor_mask
        self.classes = self.hyper_params.classes
        self.labels = self.hyper_params.labels

        ##imput img transform
        self.img_transform = transforms.Compose([
            LetterBoxImage(self.input_shape),
            transforms.ToTensor(),
            AddDim()
        ]
        )
        self.img_transform_no_addim = transforms.Compose([
            LetterBoxImage(self.input_shape),
            transforms.ToTensor()
        ]
        )
        self._post_transform = CorrectDetection(self.input_shape)

        self.model = Yolov3_abc(self.hyper_params.anchors,self.hyper_params.classes)
        self.model.load_weights(config["weights"])
        #print(config["weights"])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def detectionOneImage(self,imgname):
        ori_image = Image.open(imgname)
        img_tensor = self.img_transform(ori_image)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        features = self.model(img_tensor)

        temp_boxes = []
        for index,feature in enumerate(features):
            start = self.anchor_mask[index][0]
            end = self.anchor_mask[index][1]
            boxes = get_all_pred_boxes(feature,anchors=self.anchors[start:end],net_size=self.input_shape,num_classes=self.classes)
            temp_boxes.extend(boxes[0])
        box_array = np.array(temp_boxes,dtype=np.float32)
        nms_boxes = nms(box_array,nms_thresh=self.hyper_params.nms_thresh)
        self._post_transform(nms_boxes,ori_image.size)
        results = [ DetectionResult(box_line,ori_size=ori_image.size,labels=self.labels) for box_line in nms_boxes]
        img_a=DrawRectengle(ori_image,nms_boxes,lables=self.labels)
        return results,img_a

'''
    def evaluation(self):

        train_data_loader = torch.utils.data.DataLoader(
            VOCDetectionSet(root=self.hyper_params.data_root,
                            labels=self.hyper_params.labels,
                            data_set=self.hyper_params.data_sets,
                            transform=self.img_transform_no_addim,
                            list_file_name="test",
                            train=False
                            ),
            num_workers=4,
            batch_size=self.hyper_params.mini_batch_size, shuffle=True
        )

        recall = 0
        true_positives = 0
        false_positives = 0

        #单轮的评测

        imgpaths,img_shape,img_batch,label_target = train_data_loader.__iter__().__next__()

        if torch.cuda.is_available():
            img_batch = img_batch.cuda()
        batch_size = img_batch.data.size(0)
        features = self.model(img_batch)


        batch_outputs_dict = {i: [] for i in range(batch_size)}
        nms_boxes_dict = {}
        for index, feature in enumerate(features):
            start = self.anchor_mask[index][0]
            end = self.anchor_mask[index][1]
            boxes = get_all_pred_boxes(feature, anchors=self.anchors[start:end], net_size=self.input_shape,
                                       num_classes=self.classes)

            for batch_index, batch_boxes in enumerate(boxes):
                batch_outputs_dict[batch_index].extend(batch_boxes)

        for batch_index, yolo_boxes in batch_outputs_dict.items():
            box_array = np.array(yolo_boxes, dtype=np.float32)

            nms_boxes = nms(box_array, nms_thresh=self.hyper_params.nms_thresh)
            self._post_transform(nms_boxes, (int(img_shape[0][batch_index].data),int(img_shape[1][batch_index].data)))
            nms_boxes_dict[batch_index] = nms_boxes

        for idx in range(batch_size):
            gt_num = len(torch.nonzero(label_target[idx, :, 0]))
            batch_ground_truch = label_target[idx, :gt_num]
            batch_ground_truch=batch_ground_truch.numpy()
            DrawRectengle(Image.open(imgpaths[idx]), nms_boxes_dict[idx], lables=self.labels).save("{}.jpg".format(idx))
            dets=np.array(nms_boxes_dict[idx],dtype=np.float64)
            voc_eval(batch_ground_truch,dets)
'''






class AddDim(object):
    def __call__(self,tensor):
        '''
        单张图片的时候增加一维度
        :param tensor:
        :return:
        '''
        shape = list(tensor.shape)
        shape.insert(0, 1)
        image_ = tensor.view(tuple(shape))
        return image_

def voc_eval(labels, dets, ovthresh=0.8,inf_min=0.00001,use_07_metric=False):
    '''
    :param labels:
    :param dets:
    :param ovthresh:
    :return:
    1.将所有的det_box按det_score进行排序
    2.计算每个det_box与所有gt_box(ground-truth)的IOU
    3.取IOU最大(max_IOU)的gt_box作为这个det_box的预测结果是否正确的判断依据，然后根据max_IOU的结果判断预测结果是TP还是FP
    '''

    #center to corner
    dets[:,0] = (dets[:,0] - dets[:,2])/2
    dets[:,1] = (dets[:,1] - dets[:,3])/2
    dets[:,2] = (dets[:,2] + dets[:,0])
    dets[:,3] = (dets[:,3] + dets[:,1])

    labels[:,0] = (labels[:,0] - labels[:,2])/2
    labels[:,1] = (labels[:,1] - labels[:,3])/2
    labels[:,2] = (labels[:,2] + labels[:,0])
    labels[:,3] = (labels[:,3] + labels[:,1])



    confidence = dets[:, 4]
    BB = dets[:, 0:4]
    class_array = dets[:,5]
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind,:]


    label_mask  = np.zeros(len(labels))

    nd = len(dets)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for idx in range(nd):
        bb = BB[idx,:].astype(float)
        iou_max = -np.inf
        #计算每个det_box与所有gt_box(ground - truth)
        #的IOU
        ixmin = np.maximum(labels[:, 0], bb[0])
        iymin = np.maximum(labels[:, 1], bb[1])
        ixmax = np.minimum(labels[:, 2], bb[2])
        iymax = np.minimum(labels[:, 3], bb[3])
        iw = np.maximum(ixmax-ixmin+inf_min,0.0)
        ih = np.maximum(iymax-iymin+inf_min,0.0)
        inter_rection = iw * ih

        union =((bb[2] - bb[0] + inf_min) * (bb[3] - bb[1] + inf_min) +
                   (labels[:, 2] - labels[:, 0] + inf_min) *
                   (labels[:, 3] - labels[:, 1] + inf_min) - inter_rection)

        ious = inter_rection / union
        iou_max = np.max(ious)
        max_iou_index = np.argmax(ious)

        if iou_max > ovthresh:
            #label 要正确
            if not label_mask[max_iou_index]:
                tp[idx] = 1
                #第max_iou_index个物体有人了
                label_mask[max_iou_index]=1
        else:
            #第idx 个detection 错判
            fp[idx] = 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(len(labels))
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        #print(i)
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap