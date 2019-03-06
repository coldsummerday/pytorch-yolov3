from .monitor import Monitor


class LossMonitor(Monitor):
    stat_name = 'loss'
    #该插件的作用为简单记录每次的loss
    def _get_value(self, iteration, input, target, output, loss):
        return loss.item()