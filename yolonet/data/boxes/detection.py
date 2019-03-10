
from .box import Box




'''
yolo v3 中的detection
typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;
'''


class Detection(Box):

    def __init__(self,center_tuple,flag=True):
        """

        :param center_tuple: 坐标元组
        :param flag: True (x,y,w,h)形式 ;Flase (xmin,ymin,xmax,ymax)的形式
        """
        assert  type(center_tuple)==tuple and len(center_tuple)==4

        super(Detection,self).__init__()
        if flag:
            self.center_x,self.center_y,self.width,self.height = center_tuple
            self.center2coord()
        else:
            self.xmin,self.ymin,self.xmax,self.ymax = center_tuple
            self.coord2center()

        self.num_classes = 0
        self.prob_list = []
        self.objectness = 0.0

        #当前nms排序的是第几类
        self.sort_class = -1

    def __repr__(self):
        return "{}:[{},{},{},{},{},{}]\n".format(self.__class__.__name__,
                                               self.center_x,self.center_y,
                                               self.width,self.height,self.objectness,self.prob_list)

    """
    做nms  比较
    """
    def find_class_index(self):
        if len(self.prob_list)==0:
            return -1
        else:
            for index,prob in enumerate(self.prob_list):
                if prob !=0:
                    return index
            return -1

    def __eq__(self, other):
        assert len(self.prob_list) == len(other.prob_list)
        if other.sort_class >=0:
            return self.prob_list[other.sort_class] == other.prob_list[other.sort_class]
        else:
            return self.objectness == other.objectness
    def __lt__(self, other):
        assert len(self.prob_list) == len(other.prob_list)
        if other.sort_class >= 0:
            return self.prob_list[other.sort_class] < other.prob_list[other.sort_class]
        else:
            return self.objectness < other.objectness
    def __cmp__(self, other):
        if self < other:
            return -1
        elif self > other:
            return 1
        elif self == other:
            return 0


class DetectionResult(Box):
    def __init__(self,detection_list,ori_size,labels):
        #[x,y,w,h,conf,class_index]


        self.center_x, self.center_y, self.width, self.height,self.confidence,self.class_label_index = detection_list
        self.class_label = labels[self.class_label_index]
        width,height = ori_size
        self.center_x *= width
        self.width *=width
        self.center_y *= height
        self.height *=height

        self.center2coord()

    def __repr__(self):
        return "{}:center_x {},center_y {},width:{},height:{},confidence:{}\n".format(
            self.class_label, self.center_x, self.center_y, self.width, self.height, self.confidence
        )

    def __str__(self):
        return "{}:center_x {},center_y {},width:{},height:{},confidence:{}".format(
            self.class_label,self.center_x,self.center_y,self.width,self.height,self.confidence
        )


