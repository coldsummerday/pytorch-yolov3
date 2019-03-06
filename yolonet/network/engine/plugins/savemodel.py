#!/usr/bin/python3
from .monitor import Monitor
import torch
import os
import sys


class ModelSaver(Monitor):
    stat_name = "Savemodel"
    def __init__(self,save_path,times=[-1],prefix=None,suffix='.pkl',save_flag = False,interval=None):
        '''

        :param times:保存模型的第几个 batch 或者第几个epoch,[-1]代表每次都保存
        :param prefix:模型前缀
        :param save_flag:true 为epoch保存,false batch 保存形式
        :param interval:插件调用方式
        '''

        if interval is None:
            #如果非默认的情况下,每一轮epoch 迭代一次
            interval = [(1, 'epoch')]
        if not  save_flag:
            interval = [(1,'iteration')]
        super(ModelSaver, self).__init__(interval)
        self.times = times
        self.save_flag = save_flag
        self.save_path = save_path
        self.prefix = prefix
        self.suffix = suffix
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if self.times==[-1]:
            self.save_flag = True

    def iteration(self, *args):
        if not self.save_flag:
            batch_index = args[0]
            
            if batch_index in self.times:
                itername = str(batch_index) if batch_index != self.times[-1] else "final"
                save_name = os.path.join(self.save_path, self.prefix + "_" + itername + self.suffix)
                self.save(save_name)

    def epoch(self, epoch_idx):
        if  self.save_flag:
            if self.times == [-1]:
                itername = str(epoch_idx) if epoch_idx != self.times[-1] else "final"
                save_name = os.path.join(self.save_path, self.prefix + "_" + itername + self.suffix)

                self.save(save_name)
            else:
                if epoch_idx in self.times:
                    itername = str(epoch_idx) if epoch_idx != self.times[-1] else "final"
                    save_name = os.path.join(self.save_path, self.prefix + "_" + itername + self.suffix)
                    self.save(save_name)

    def save(self, name):
        self.trainer.model.save_weights(name)
        #torch.save(self.trainer.model.state_dict(),
        #           name)
        '''
        if isinstance(self.trainer.model, Darknet):
            self.trainer.model.save_weights(name)
        else:
            torch.save(self.trainer.model.state_dict(),
                       name)
        '''



