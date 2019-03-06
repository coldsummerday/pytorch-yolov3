import logging as log
import  torch
import torch.optim as optim
from  collections import  OrderedDict
import os
this_file_path = os.path.abspath(os.path.dirname(__file__))
from ...data.dataset import VOCDetectionSet
import torchvision.transforms as transforms
from ..module import Yolov3_abc
from ...hyperparams import HyperParams
from .trainer import Trainer
from  .plugins import *
from ..loss import  YoloLoss

__all__=["VOCTrianningEngine"]

class VOCTrianningEngine(object):

    def __init__(self,config):

        hyper_params = HyperParams(config)

        #all args
        self.batch_size = hyper_params.batch_size
        self.mini_batch_size = hyper_params.mini_batch_size
        learning_rate = config['warmup_lr']
        momentum = config['momentum']
        decay = config['decay']
        self.lr_steps = config['lr_steps']
        self.lr_rates = config['lr_rates']

        self.model = Yolov3_abc(config)
        if torch.cuda.is_available():
            self.model.cuda()


        self.model.load_weights(config["weights"])


        train_data_loader = torch.utils.data.DataLoader(
        VOCDetectionSet(root=hyper_params.data_root,
                        labels=hyper_params.labels,
                        data_set=hyper_params.data_sets,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]),
                        batch_size=self.mini_batch_size,
                        ),
            batch_size=self.mini_batch_size,shuffle=True
        )
        self.batch_scale = int(hyper_params.batch_size / hyper_params.mini_batch_size)
        
        optim = torch.optim.SGD(self.model.parameters(),
                                lr=learning_rate / self.batch_size, momentum=momentum, dampening=0,
                                weight_decay=decay * self.batch_size)
        self.trainer = YoloTrainner(model=self.model,dataset=train_data_loader,optimizer=optim)

        self.trainer.register_plugin(TimeMonitor())
        self.trainer.register_plugin(LossMonitor())
        self.trainer.register_plugin(LrAdjust(lr_steps=self.lr_steps,lr_rates=self.lr_rates))
        self.trainer.register_plugin(Logger(['loss', 'time']))
        save_path = os.path.join(this_file_path,"../../../",config["output_root"],config["output_version"])
        save_mkdir(save_path)
        save_times = [i*config["backup_steps"][0] for i in range(10)]
        save_times.extend([i *config["backup_rates"][0]  for i in range(1,config["max_batches"]//config["backup_rates"][0])])
        self.trainer.register_plugin(ModelSaver(save_path=save_path,prefix=config["output_version"],suffix=".weights",times=save_times))
        self.trainer.run(len(train_data_loader)//self.mini_batch_size+1)


class YoloTrainner(Trainer):
    def __init__(self, model=None, optimizer=None, dataset=None, USE_CUDA=True,batch_scale =8):
        super(YoloTrainner,self).__init__(model=model,optimizer=optimizer,dataset=dataset,USE_CUDA=USE_CUDA)
        self.loss_fn_list = []
        self.batch_scale = batch_scale
        for idx,yolo_info in enumerate(self.model.yolo_infos):
            self.loss_fn_list.append(YoloLoss(self.model.num_classes,anchors=yolo_info.anchors,
                                              anchors_mask=yolo_info.anchors_mask,
                                              reduction=yolo_info.reduction,head_idx=idx))
                                            
    def train(self):
        for i, data in enumerate(self.dataset, self.iterations + 1):
            batch_input, batch_target = data
            #在每次获取batch data 后进行更新
            self.call_plugins('batch', i//self.batch_scale, batch_input, batch_target)
            input_var = batch_input
            target_var = batch_target
            if self.USE_CUDA:
                input_var = input_var.cuda()
                target_var = target_var.cuda()
            #这里是给后续插件做缓存部分数据,这里是网络输出与loss
            plugin_data = [None, None]
            def closure():
                features = self.model(input_var)
                assert  len(features) == len(self.loss_fn_list)
                loss_list = []
                for idx in range(len(features)):
                    loss_list.append(self.loss_fn_list[idx](features[idx],target_var))
                loss = sum(loss_list)
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = features
                    plugin_data[1] = loss.data
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration',  i//self.batch_scale, batch_input, batch_target,
                              *plugin_data)
            self.call_plugins('update',  i//self.batch_scale, self.model)

        self.iterations += i

def save_mkdir(path):
    if  not os.path.exists(path):
        os.mkdir(path)

