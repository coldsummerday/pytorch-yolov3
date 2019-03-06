

__all__=["Box"]

class Box(object):

    def __init__(self):
        self.class_label = ""
        self.object_id = 0
        self.center_x = 0.0
        self.center_y = 0.0
        self.width = 0.0
        self.height = 0.0

        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0

    def center2coord(self):
        self.xmin = self.center_x - self.width / 2
        self.xmax = self.center_x + self.width / 2
        self.ymin = self.center_y - self.height / 2
        self.ymax = self.center_y + self.height / 2


    def coord2center(self):
        self.center_x = (self.xmin + self.xmax) / 2
        self.center_y = (self.ymin + self.ymax) / 2
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin



