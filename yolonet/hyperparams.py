import logging as log
import torch

__all__ = ['HyperParams']

class HyperParams(object):
    def __init__(self,config,train_flag = 1):
        self.cuda = True
        self.yolo_layer_cuda = False
        self.labels = config['labels']
        self.classes = len(self.labels)
        self.data_root = config['data_root_dir']
        self.data_sets = config["sets"]
        self.model_name = config['model_name']
        self.net_width, self.net_height = config['input_shape']
        self.anchors =parse_anchor_str(config["anchors"])
        self.anchor_mask = parse_anchor_mask(config["anchors_mask"])

        self.conf_thresh = 0.5

        if self.cuda:
            if not torch.cuda.is_available():
                log.debug('CUDA not available')
                self.cuda = False
            else:
                log.debug('CUDA enabled')

        ##test 时候evaluation 用到
        self.batch_size  = config['batch_size']
        self.mini_batch_size = config['mini_batch_size']
        ##train
        if train_flag==1:
            #lr control
            self.lr_steps = config["lr_steps"]
            self.lr_rates = config["lr_rates"]



            self.max_batches = config["max_batches"]

            #backup control
            self.backup_interval=config["backup_interval"]
            self.backup_steps = config["backup_steps"]
            self.backup_rates = config["backup_rates"]

            #cal yolo loss
            self.ignore_thresh = 0.5
            self.truth_thresh = 1.0
            self.rescore = 0
            self.max_boxes = 90

        else:

            #nms
            self.nms_thresh = 0.4






def parse_anchor_mask(mask_list):
    result = []
    for mask_str in mask_list:
        elements = mask_str.split(',')
        start = int(elements[0])
        end = int(elements[1])
        result.append((start,end))
    return result



def parse_anchor_str(anchor_line):
    anchors = anchor_line.split(",")
    anchors = [int(float(a)) for a in anchors]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    return anchors


