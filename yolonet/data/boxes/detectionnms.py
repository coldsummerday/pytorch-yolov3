
'''
yolo 源码中的nms
int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    //diff < 0 则a < b
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}
qsort如果其第一个参数比第二个参数小，则返回一个小于0的值，反之则返回一个大于0的值，如果相等，则返 回0。
void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}
'''
def detect_box_nms_sort(detectBoxs,thresh,num_class):
    num_boxes = len(detectBoxs)

    for class_index in range(num_class):
        for i in range(num_boxes):
            detectBoxs[i].sort_class = class_index
        #从大到小排列
        detectBoxs =  sorted(detectBoxs,reverse=True)
        for i in range(num_boxes):
            if (detectBoxs[i].prob_list[class_index]==0):
                continue

            for j in range(i+1,num_boxes):
                if detectBoxs[j].objectness == 0:
                    continue

                boxs_iou = bbox_iou_boxclass(detectBoxs[i],detectBoxs[j])
                if boxs_iou > thresh:
                    detectBoxs[j].objectness = 0
                    for k in range(num_class):
                        detectBoxs[j].prob_list[k] = 0
    new_detectBoxes = []
    for detectbox in detectBoxs:
        if detectbox.objectness > thresh:
            new_detectBoxes.append(detectbox)
    return new_detectBoxes

def bbox_iou_boxclass(boxA,boxB):
    assert  type(boxA)==type(boxB)
    a_x1 = boxA.center_x - boxA.width / 2.0
    a_y1 = boxA.center_y - boxA.height / 2.0
    a_x2 = boxA.center_x + boxA.width / 2.0
    a_y2 = boxA.center_y + boxA.height / 2.0

    b_x1 = boxB.center_x - boxB.width / 2.0
    b_y1 = boxB.center_y - boxB.height / 2.0
    b_x2 = boxB.center_x + boxB.width / 2.0
    b_y2 = boxB.center_y + boxB.height / 2.0

    # 相交部分
    inter_rect_x1 = max(a_x1, b_x1)
    inter_rect_y1 = max(a_y1, b_y1)
    inter_rect_x2 = min(a_x2, b_x2)
    inter_rect_y2 = min(a_y2, b_y2)

    # 相交面积
    inter_area = clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # 各自原来面积
    boxa_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    boxb_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    iou = inter_area / (boxa_area + boxb_area - inter_area + 0.000001)
    return iou

def clamp(x,min=0,max=1):
    if x <= min:
        return 0
    elif  x > min and x < max:
        return x
    elif x>=max:
        return max