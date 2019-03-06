import heapq
import logging

class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None,USE_CUDA=True,max_batch=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.USE_CUDA = USE_CUDA
        #总的迭代次数
        self.iterations = 0
        self.stop_flag = False
        self.max_batch =max_batch   if max_batch else 50020
        '''
        Trainer的状态，注意这里的状态包含了所有插件提供的状态。初始化为空
        '''
        self.stats = {}

        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }
        '''
        作者将插件的调用进行了分类:
        (1)iteration:一般是在完成一个batch 训练之后进行的事件调用序列（一般不改动网络或者优化器，如：计算准确率）调用序列；
        (2)batch 在进行batch 训练之前需要进行的事件调用序列
        (3)epoch 完成一个epoch 训练之后进行的事件调用序列
        (4)update 完成一个batch训练之后进行的事件(涉及到对网络或者优化器的改动,如:学习率的调整)
        
        iteration 跟update 两种插件调用的时候传入的参数不一样,iteration 会传入batch output,loss 等训练过程中的数据,
        而update传入的的model ,方便对网络的修改
        '''

    def register_plugin(self, plugin):
        #注册插件
        plugin.register(self)

        #插件的触发间隔,一般是这样的形式[(1, 'iteration'), (1, 'epoch')]
        intervals = plugin.trigger_interval

        if not isinstance(intervals, list):
            intervals = [intervals]
        for duration, unit in intervals:
            #unit 是事件的触发类别
            queue = self.plugin_queues[unit]
            '''添加事件， 这里的duration就是触发间隔,，以后在调用插件的时候，
            会进行更新  duration 决定了比如在第几个iteration or epoch 触发事件。len(queue)这里应当理解为优先级（越小越高）
            【在相同duration的情况下决定调用的顺序】，根据加入队列的早晚决定。'''
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        #调用插件
        args = (time,) + args
        #这里的time 最基本的意思是次数,如(iteration or epoch)
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            '''如果队列第一个事件的duration（也就是触发时间点）小于当前times'''
            plugin = queue[0][2]
            '''调用相关队列相应的方法，所以如果是继承Plugin类的插件，
                       必须实现 iteration、batch、epoch和update中的至少一个且名字必须一致。'''
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            '''根据插件的事件触发间隔，来更新事件队列里的事件 duration'''
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)
            '''加入新的事件并弹出最小堆的堆头。最小堆重新排序。'''

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            '''对四个事件调用序列进行最小堆排序。'''
            heapq.heapify(q)

        for i in range(1, epochs + 1):
            if  self.stop_flag:
                logging.warn("stop trainning!")
                break
            self.train()
            #进行每次epoch 的更新
            self.call_plugins('epoch', i)

    def train(self):
        for i, data in enumerate(self.dataset, self.iterations + 1):
            if self.stop_flag or self.iterations > self.max_batch:
                logging.warn("stop trainning!")
                break
            batch_input, batch_target = data
            #在每次获取batch data 后进行更新
            self.call_plugins('batch', i, batch_input, batch_target)
            input_var = batch_input
            target_var = batch_target
            if self.USE_CUDA:
                input_var = input_var.cuda()
                target_var = target_var.cuda()
            #这里是给后续插件做缓存部分数据,这里是网络输出与loss
            plugin_data = [None, None]

            def closure():

                loss = self.model(input_var,target_var)
                #yolo train输出的时候已经为loss
                #loss = self.criterion(batch_output, target_var)
                loss.backward()
                if plugin_data[0] is None:
                    #plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', i, batch_input, batch_target,
                              *plugin_data)
            self.call_plugins('update', i, self.model)

        self.iterations += i

    def stop(self):
        self.stop_flag = True



