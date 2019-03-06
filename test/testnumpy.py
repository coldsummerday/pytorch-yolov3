import  numpy as np
from yolonet.data.boxes.detection import Detection
import  sys
import  time
sys.path.insert(0,"/home/zhou/code/cpython_demo/cmodule/")
import pickle
f = open('detections.pkl', 'rb')
data = pickle.load(f)
import nms


def detection2numpy(detections):
    pre_list = []
    for detection in detections:
        line = [detection.center_x,detection.center_y,detection.width,detection.height,detection.objectness]
        line.extend(detection.prob_list)
        pre_list.append(line)
    np_array =  np.array(pre_list, dtype=np.float32)
    return np_array

start = time.time()

np_array = detection2numpy(data)
result = nms.cpu_nms(np_array,0.45,10)
end = time.time()
print("c lib nms spend time {}".format(end-start))
print(np_array.shape)
print(result)
print(len(result))


