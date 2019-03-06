from .monitor import Monitor
import torch
import  logging as log

class LrAdjust(Monitor):
    stat_name = "LrAdjust"
    def __init__(self,lr_steps,lr_rates):
        #每5000次batch 调用一次
        interval = [(100, 'iteration')]
        super(LrAdjust,self).__init__(interval)

        self.lr_steps = lr_steps
        self.lr_rates = lr_rates
        self.length = len(self.lr_steps)
        self.lr_index = 0
    def iteration(self, *args):
        batch_index = args[0]
        if self.lr_index < self.length and batch_index == self.lr_steps[self.lr_index]:
            lr = self.lr_rates[self.lr_index]
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = lr
                log.debug("model leanring rate change to {}".format(lr))
            self.lr_index += 1
