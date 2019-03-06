import os
import logging as log
import torch
import torch.nn as nn
from ._darknet_load import DarknetLoader
from ..layers import _darknet_layer as net_layer
from .utils import parse_cfg

this_file_path  = os.path.abspath(os.path.dirname(__file__))
from ...hyperparams import HyperParams



class Yolov3_abc(DarknetLoader):
    """
    用darknet的cfg文件去创建网络结构,
    本yolo 抽象类完成 网络骨干的搭建
    需要知道 识别的种类,所以需要env
    """
    def __init__(self,cfg_dict):
        """ Network initialisation """
        super().__init__()
        # Parameters

        self.hyperparams = HyperParams(config=cfg_dict)
        self.num_classes = self.hyperparams.classes

        self.yolo_infos = []


        self.create_modules(os.path.join(this_file_path,"../darknet_cfg/yolov3.cfg"))


    def create_modules(self,cfg_file):
        self.blocks = parse_cfg(cfg_file)
        self.cache_index = []
        self.module_list = nn.ModuleList()

        index = 0
        prev_filters = 3
        output_filters = []
        # 第一个block 为net 信息,跳过
        for x in self.blocks:
            module = nn.Sequential()
            if (x["type"] == "net"):
                continue

            if(x["type"]=="convolutional"):

                if "batch_normalize" in x.keys():
                    batch_normalize = True
                    bias = False
                else:
                    batch_normalize = False
                    bias = True
                filters = int(x.get("filters",32)) # int(x["filters"])
                padding = x.get("pad",0) # int(x["pad"])
                kernel_size = int(x.get("size",3)) # int(x["size"])
                stride = int(x.get("stride",1)) #int(x["stride"])



                if padding:
                    pad = int(kernel_size / 2)
                else:
                    pad = 0
                if batch_normalize:
                    conv_batch = net_layer.Conv2dBatchLeaky(prev_filters,filters,kernel_size,stride,pad)
                    module.add_module("conv2dBatchleaky_{0}".format(index),conv_batch)
                else:
                    #此处filters 是用来提取种类识别信息  filters = 3*(classes + 5)
                    conv = nn.Conv2d(prev_filters, 3 *(self.num_classes + 5), kernel_size, stride, pad, bias=bias)
                    module.add_module("conv_{0}".format(index), conv)



            elif (x["type"] == "route"):
                layer_indexs = []
                x["layers"] = x["layers"].split(",")
                start = int(x["layers"][0])
                if len(x["layers"]) == 2:
                    end = int(x["layers"][1])
                else:
                    end = 0
                if start > 0:
                    start= start - index
                if end > 0:
                    end = end - index

                truth_start_index = index + start
                layer_indexs.append(truth_start_index)
                if end < 0:
                    #需要短接两个层的部分
                    truth_end_index = index + end
                    layer_indexs.append(truth_end_index)
                    filters = output_filters[truth_start_index] + output_filters[truth_end_index]
                else:
                    #只需要短接一个层
                    filters = output_filters[truth_start_index]

                self.cache_index.extend(layer_indexs)

                route_layer = net_layer.RouteLayer(index,layer_indexs)
                module.add_module("Route_{0}".format(index), route_layer)


            elif (x["type"] == "upsample"):
                upsample = net_layer.InterpolateUpsample(scale_factor=2, mode="nearest")
                module.add_module("upsample_{}".format(index), upsample)


            elif x["type"] == "shortcut":
                from_ = int(x["from"])
                shortcut_layer = net_layer.ShortcutLayer(index,from_)
                module.add_module("shortcut_{}".format(index), shortcut_layer)
                self.cache_index.extend([index-1,index+from_])


            elif x["type"] == "maxpool":
                stride = int(x["stride"])
                size = int(x["size"])
                if stride != 1:
                    maxpool_layer = nn.MaxPool2d(size, stride)
                else:
                    maxpool_layer = net_layer.MaxPoolStride1(size)
                module.add_module("maxpool_{}".format(index), maxpool_layer)



            elif x["type"] == "yolo":
                mask = x["mask"].split(",")
                mask = tuple([int(x) for x in mask])

                yolo_scale =int(x["scale"])
                yolo_info_layer = net_layer.YoloLayerInfo(self.hyperparams.anchors,mask,reduction=yolo_scale)
                self.yolo_infos.append(yolo_info_layer)
                module.add_module("yolo_{}".format(index), net_layer.EmptyLayer())
            else:
                log.debug("skip add layer {}".format(x["type"]))
                assert False
            self.module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)
            index += 1
    def forward(self, x,target=None):
        modules = self.blocks[1:]
        layer_outputs = {}
        features = []
        for i in range(len(modules)):
            module_type = modules[i]["type"]
            if module_type in ["convolutional","upsample","maxpool"]:
                x = self.module_list[i](x)

            # route 层跟 shorcut 层都需要对 之前的层做操作,所以需要传入output
            elif module_type == "route":
                x = self.module_list[i](layer_outputs)
            elif module_type == "shortcut":
                #resnet 结构
                x = self.module_list[i](layer_outputs)
            elif module_type == "yolo":
                features.append(x)


            if i in self.cache_index:
                layer_outputs[i] = x

        return features




