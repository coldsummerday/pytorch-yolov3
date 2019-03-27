# pytorch-yolov3
Implement  yolo-v3 by use pytorch for one-stage object detection

##Requirements
python 3.6
pytorch 0.4.0
opencv-python

##installtion


```
git@github.com:coldsummerday/pytorch-yolov3.git
cd pytorch-yolov3/
pip3 install -r requirements.txt
yolo_root=$(pwd)
cd ${yolo_root}/yolonet/data/transforms/yolonms
make 
```

##detector

downlowd  yolov3.weights from darknet website:

```
wget https://pjreddie.com/media/files/yolov3.weights
```

 move  the model weights to ${yolo_root}/data directory.
 
 ```
python3 example/yolo_detect.py --image path/to your image
 ```
 
 
##train

```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar

wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

tar xf VOCtrainval_11-May-2012.tar

tar xf VOCtrainval_06-Nov-2007.tar

tar xf VOCtest_06-Nov-2007.tar
cd VOCdevkit
VOCdevkit_root=$(pwd)
cd ${yolo_root}
```

change the  cfg/yolov3.yml 


```
data_root_dir: "{VOCdevkit_root}"
sets:
 - "VOC2007"
 - "VOC2012"
 ```
 
 download the pretrain model:
 ```
 wget https://pjreddie.com/media/files/darknet53.conv.74
 ```
 
 move the darknet53.conv.74 to ${yolo_root}/data
 
 Then   start train!
 
 
 ```
 python3 example/yolo_train.py
 ```
 
 
 
##Calculate the anchors 

```
python3 example/calculateAnchor.py -s  416
```

the output will be the follow:
```
2019-03-27 20:23:35,374:DEBUG:---------start collect annotations-------
2019-03-27 20:23:43,335:DEBUG:collect 142178 boxes,take 7.961443 second
2019-03-27 20:23:43,336:DEBUG:---------start kmeans-------
2019-03-27 20:24:06,280:DEBUG:kmeans end! takes 22.944322 second
2019-03-27 20:24:06,280:DEBUG:result is  
 [[ 27.08333333  42.75555556]
 [ 25.35        50.84444444]
 [ 29.46666667  48.91851852]
 [ 24.05        61.24444444]
 [ 33.8         44.68148148]
 [ 28.16666667  59.31851852]
 [ 34.01666667  56.23703704]
 [ 26.86666667  76.26666667]
 [ 78.         104.77037037]]
2019-03-27 20:24:07,862:DEBUG:the dataset iou is 0.8878912100707463 
```

#to do:

[1] eval
[2]Distributed training
[3] more backbone for yolo v3

 
