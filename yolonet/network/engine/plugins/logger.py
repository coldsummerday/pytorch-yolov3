from collections import defaultdict
from .plugin import Plugin
import logging

class Logger(Plugin):
    alignment = 4
    #不同字段之间的分隔符
    separator = '#' * 80

    def __init__(self, fields, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Logger, self).__init__(interval)

        #需要打印的字段,如loss acc
        self.field_widths = defaultdict(lambda: defaultdict(int))
        self.fields = list(map(lambda f: f.split('.'), fields))
        # 遵循XPath路径的格式。以AccuracyMonitor为例子，如果你想打印所有的状态，
        # 那么你只需要令fields=[AccuracyMonitor.stat_name]，也就是，['accuracy']，
        # 而如果你想只打印AccuracyMonitor的子状态'last'，那么你就只需要设置为
        # ['accuracy.last'],而这里的split当然就是为了获得[['accuracy', 'last']]
        # 这是为了之后的子状态解析（类似XPath路径解析）所使用的。

    def _join_results(self, results):
        # 这个函数主要是将获得的子状态的结果进行组装。
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def log(self, msg):
        logging.debug(msg)

    def register(self, trainer):
        self.trainer = trainer

    def gather_stats(self):
        result = {}
        return result

    def _align_output(self, field_idx, output):
        #对其输出格式
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        # 这个函数是核心，负责将查找到的最底层的子模块的结果提取出来。
        output = []
        name = ''
        if isinstance(stat, dict):
            '''
            通过插件的子stat去拿到每一轮的信息,如LOSS等
            '''
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            # 找到自定义的输出名称。y有时候我们并不像打印对应的Key出来，所以可以
            # 在写插件的时候增加多一个'log_name'的键值对，指定打印的名称。默认为
            # field的完整名字。传入的fileds为['accuracy.last']
            # 那么经过初始化之后，fileds=[['accuracy',
            # 'last']]。所以这里的'.'.join(fields)其实是'accuracy.last'。
            # 起到一个还原名称的作用。
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            # 在这里的话，如果子模块stat不是字典且require_dict=False
            # 那么他就会以父模块的打印格式和打印单位作为输出结果的方式。
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output

    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        loginfo = []

        if prefix is not None:
            loginfo.append(prefix)
            loginfo.append("\t")

        loginfo.append(output)
        if suffix is not None:
            loginfo.append("\t")
            loginfo.append(suffix)
        self.log("".join(loginfo))

    def iteration(self, *args):
        '''
        :param args:   ( i, batch_input, batch_target,*plugin_data) 的元祖
        :return:
        '''
        self._log_all('log_iter_fields',prefix="iteration:{}".format(args[0]))

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields',
                      prefix=self.separator + '\nEpoch summary:',
                      suffix=self.separator,
                      require_dict=True)