
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b



def cpu_nms(np.ndarray[np.float32_t,ndim=2] dets,np.float  nms_thresh,int num_classes):
    cdef  np.ndarray[np.float32_t,ndim=1] x = dets[:,0]
    cdef  np.ndarray[np.float32_t,ndim=1] y = dets[:,1]
    cdef  np.ndarray[np.float32_t,ndim=1] w = dets[:,2]
    cdef  np.ndarray[np.float32_t,ndim=1] h = dets[:,3]
    cdef  np.ndarray[np.float32_t,ndim=1] confidence = dets[:,4]
    cdef  np.ndarray[np.float32_t,ndim=2] prob = dets[:,5:]

    cdef  np.ndarray[np.float32_t,ndim=1] xmin = x - w / 2
    cdef  np.ndarray[np.float32_t,ndim=1] xmax = x + w / 2
    cdef  np.ndarray[np.float32_t,ndim=1] ymin = y - h / 2
    cdef  np.ndarray[np.float32_t,ndim=1] ymax = y + h / 2


    cdef np.ndarray[np.float32_t, ndim=1] areas =w * h
    cdef np.ndarray[np.int_t, ndim=1] sort_order
    cdef int ndets = dets.shape[0]

    cdef np.float32_t min_value = 0.00001

    cdef np.float32_t inter, iou

    cdef np.float32_t inter_rect_x1,inter_rect_y1,inter_rect_x2,inter_rect_y2
    cdef np.float32_t inter_rect_w,inter_rect_h
    cdef int i,j,k,class_index,truth_i_index,truth_j_index

    result = []

    for class_index in range(num_classes):
        sort_order = prob[:,class_index].argsort()[::-1]
        for i in range(num_classes):
            truth_i_index = sort_order[i]
            iarea = areas[truth_i_index]
            if prob[truth_i_index,class_index]==0:
                    continue
            for j in range(i+1,ndets):
                truth_j_index = sort_order[j]
                if prob[truth_j_index,class_index]==0:
                    continue
                 # 相交部分
                inter_rect_x1 = max(xmin[truth_i_index], xmin[truth_j_index])
                inter_rect_y1 = max(ymin[truth_i_index], ymin[truth_j_index])
                inter_rect_x2 = min(xmax[truth_i_index], xmax[truth_j_index])
                inter_rect_y2 = min(ymax[truth_i_index], ymax[truth_j_index])

                inter_rect_w = max(0.0,inter_rect_x2 - inter_rect_x1 + min_value)
                inter_rect_h = max(0.0,inter_rect_y2 - inter_rect_y1 + min_value)

                inter = inter_rect_w * inter_rect_h

                iou = inter / (areas[truth_i_index] + areas[truth_j_index] - inter)

                if iou > nms_thresh:
                    prob[truth_j_index,class_index] = 0
    for i in range(ndets):
        for j in range(num_classes):
            if prob[i,j] > 0:
                result.append([x[i],y[i],w[i],h[i],prob[i,j],j])
    return result