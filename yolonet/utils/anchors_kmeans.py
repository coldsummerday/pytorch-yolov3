import os

import numpy as np
from ..data.dataset import PascalVocReader
from ..data.dataset.PascalVocxml import ParserxmlError
from .env_config import initEnv
import logging





def kmeans(boxes,k=9,dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows   (w,h)
    :param k: number of clusters
    :param dist: distance function,这次为取聚集点中的中位数
    :return: numpy array of shape (k, 2)

    """
    
    rows = boxes.shape[0]
    
    #每一个框与 k个中心点的距离
    distances = np.empty((rows,k))
    
    last_clusters = np.zeros((rows,))
    np.random.seed()

    #随机选9个点作为初始点
    clusters = boxes[np.random.choice(rows,k,replace=False)]
    while True:
        for row in range(rows):
            #距离函数
            distances[row] = 1 - kmeans_iou(boxes[row],clusters)

        neareset_clusters = np.argmin(distances,axis=1)

        if(last_clusters==neareset_clusters).all():
            break
        for index_clu in range(k):
                #在与第K个中心点最近的点中选择新的点作为局簇中心点
                #默认distance 选用中位数,也就是离第K个中心点 最近的这些点的中位数作为新的点,所以要先做一波排序
            optional_boxes = sort_anchors(boxes[neareset_clusters==index_clu])
            clusters[index_clu] = dist(optional_boxes,axis=0)
        last_clusters = neareset_clusters
    return sort_anchors(clusters)
    
def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(kmeans_iou(boxes[i], clusters)) for i in range(boxes.shape[0])])
            
def sort_anchors(boxs):
    '''
    根据w * h 的面积来排序 box,升序
    :param box: np.array((r,2)),r个2维数组,
    return numpy array of shape(k,2)
    '''
    area_array  = boxs[:,0] * boxs[:,1]
    index = area_array.argsort()
    return boxs[index]

    

    

def kmeans_iou(box,clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    计算一个框 与  K个聚合点的iou 
    因为kmeans 的时候,只计算 w,h(只计算宽高anchors)
    """
    min_w = np.minimum(clusters[:,0],box[0])
    min_h = np.minimum(clusters[:,1],box[1])
    
    if np.count_nonzero(min_w==0) > 0 or np.count_nonzero(min_h==0) > 0:
        raise ValueError("Box has no area")
    intersection_area = min_w * min_h
    
    box_area = box[0] * box[1]
    clusters_area = clusters[:,0] * clusters[:,1]
    iou_ = intersection_area / (box_area + clusters_area - intersection_area)
    
    return iou_
    




    
    
