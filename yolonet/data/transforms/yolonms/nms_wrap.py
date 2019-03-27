from .nmsclib import cpu_nms
import numpy as np

def nms(ndarray,nms_thresh):
    assert type(ndarray)==np.ndarray
    assert type(nms_thresh)==float
    print(ndarray)
    if len(ndarray)==0:
        return []
    ndim = ndarray.shape[1]
    classes_num = ndim - 5
    result = cpu_nms(ndarray,nms_thresh,classes_num)
    
    return result

