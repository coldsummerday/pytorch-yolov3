

from ..boxes.detection import Detection

from PIL import Image,ImageDraw


class CorrectDetection:
    def __init__(self,net_size):

        self.net_size = net_size

    def __call__(self,detections,img_size):
        """
        nms 后的每条记录为[x,y,w,h,confidence,classid]
        :param detections:包含每条记录的list
        :param img_size: 包含原来图像的元组(img_w,img_h)
        :return:
        """
        img_width, img_height = img_size
        net_height,net_width = self.net_size

        if net_width / img_width < net_height / img_height:
            new_width = net_width
            new_height = (img_height * net_width) / img_width
        else:
            new_width = (img_width * net_height) / img_height
            new_height = net_height
        x_old, x_slide = (new_width - new_width) / (2 * net_width), net_width / new_width
        y_old, y_slide = (net_height - new_height) / (2 * net_height), net_height / new_height
        for i in range(len(detections)):
            if detections[i][4] == 0:
                continue
            detections[i][0] = (detections[i][0] - x_old) * x_slide
            detections[i][1] = (detections[i][1] - y_old) * y_slide
            detections[i][2] *= x_slide
            detections[i][3] *= y_slide
        return detections


def DrawRectengle(ori_img,detections,lables):

    ori_width,ori_height = ori_img.size
    draw_handle = ImageDraw.Draw(ori_img)
    for box_line in detections:
        x, y, w, h, conf,label_index = box_line
        xmin = int( (x-w/2) * ori_width)
        ymin = int((y-h/2) * ori_height)
        xmax = int((x + w/2) *ori_width)
        ymax = int((y + h/2 ) * ori_height)
        label = lables[label_index]
        draw_handle.rectangle([(xmin,ymin),(xmax,ymax)])
        draw_handle.text((xmin, ymin-10), label)
    return ori_img




