import os
import argparse
import numpy as np
from yolonet.data.dataset import PascalVocReader,ParserxmlError
from yolonet.utils.env_config import initEnv
from yolonet.utils.anchors_kmeans import kmeans,avg_iou
import logging
import time
logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
         level=logging.DEBUG)
log = logging.getLogger(__name__)

def arg_parse():
    '''
    Parse arguements
    '''
    parser = argparse.ArgumentParser(description="YOLO v3  kmeans to anchor w,h")

    parser.add_argument('-s', "--size", dest="size", help="network size", type=int,default=416)

    parser.add_argument("-m","--model", dest="modelname", help="model name  to choise the yaml file,default is Yolov3", default="Yolov3")
    
    return parser.parse_args()
    
def getShapes(files):
    shapes = []
    for filename in files:
        try:
            voc_xml_handler = PascalVocReader(filename)
            voc_xml_handler.setPercent(True)
            voc_xml_handler.parse()
        except ParserxmlError:
            logging.warning('skip file {}'.format(filename))
            continue
        shapes.extend(voc_xml_handler.getShapes())
    return shapes
    
def load_voc_annotation_w_h(modelname):
    config = initEnv(2,modelname)
    data_root_dir = config['data_root_dir']
    annotation_files = []
    for dataSet in config['sets']:
        annotation_path = os.path.join(data_root_dir,str(dataSet)+"/Annotations/")
        annotations = os.listdir(annotation_path)
        annotations = [os.path.join(annotation_path,filename) for filename in annotations]
        annotation_files.extend(annotations)
    object_shapes = getShapes(annotation_files)
    w_h_list = []
    for shape in object_shapes:
        points = shape[1]
        #[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        w = points[1][0] - points[0][0]
        h = points[2][1] - points[0][1]
        w_h_list.append([w,h])
    return np.array(w_h_list)

def  main():
    
    args = arg_parse()
    model_name = args.modelname
    network_size = args.size
    start_time = time.time()
    log.debug("---------start collect annotations-------")
    annotation_array = load_voc_annotation_w_h(model_name)
    collect_time = time.time()
    log.debug("collect %d boxes,take %f second" %(annotation_array.shape[0],collect_time-start_time))
    log.debug("---------start kmeans-------")
    kmeans_out = kmeans(annotation_array,k=9) 
    kmeans_time = time.time()
    log.debug("kmeans end! takes %f second" %(kmeans_time-collect_time))
    log.debug("result is  \n %s" %(str(kmeans_out * network_size)))
    iou = avg_iou(annotation_array,kmeans_out)
    log.debug("the dataset iou is %s " %(iou))
if __name__=="__main__":
    main()