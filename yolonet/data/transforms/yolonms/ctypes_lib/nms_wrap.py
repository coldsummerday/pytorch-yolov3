from ctypes import *
import os

this_file_path = os.path.abspath(os.path.dirname(__file__))
lib = CDLL(os.path.join(this_file_path,"libnms.so"), RTLD_GLOBAL)

__all__=["nms"]


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]


def list2FloatPoint(param):
    val = ((c_float)*len(param))(*param)
    return val

def detections2PointDe(detections):
    results = []
    for detect in detections:
        temp_c_detect = DETECTION()
        temp_c_detect.bbox = BOX(detect.center_x,detect.center_y,detect.width,detect.height)
        temp_c_detect.classes = detect.num_classes
        temp_c_detect.prob = list2FloatPoint(detect.prob_list)
        temp_c_detect.objectness = detect.objectness
        temp_c_detect.sort_class = detect.sort_class
        results.append(temp_c_detect)
    return (DETECTION * len(results))(*results)

def nms(detections,num_classes,nms_thresh):
    length = len(detections)
    dets = detections2PointDe(detections)
    do_nms_sort(dets, c_int(length), c_int(num_classes), c_float(nms_thresh));
    result = []
    for j in range(length):
        for i in range(10):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                ##x,y,w,h,confidence,id
                result.append((b.x,b.y,b.w,b.h,dets[j].prob[i],i))
    return result
                

    
    
        
    
    
