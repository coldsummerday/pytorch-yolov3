
import pickle
f = open('detections.pkl', 'rb')
data = pickle.load(f)
from yolonet.data.boxes.detectionnms import detect_box_nms_sort
from nms_wrap import *
import  time
start = time.time()
nms(data)
end = time.time()
print("c lib nms spend time {}".format(end-start))

start = time.time()
detect_box_nms_sort(data,0.45,10)
end = time.time()
print("pure python spend time {}".format(end-start))











"""
$ python3 test.py
c lib nms spend time 0.0002491474151611328
pure python spend time 0.0004749298095703125


"""