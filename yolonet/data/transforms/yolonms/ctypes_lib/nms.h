#ifndef __NMS_H
#define __NMS_H
typedef struct{
    float x, y, w, h;
} box;
typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;
float box_iou(box a, box b);
void do_nms_sort(detection *dets, int total, int classes, float thresh);
#endif

