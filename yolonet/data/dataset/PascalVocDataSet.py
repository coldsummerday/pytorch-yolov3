#!/usr/bin/python3


import  random
from torch.utils.data import  Dataset
import os
from  .dataUtils import load_data_detection
from torchvision import  transforms
pil2tensor_transforms = transforms.Compose([
                            transforms.ToTensor(),
                        ])

__all__=["VOCDetectionSet"]

class VOCDetectionSet(Dataset):
    def __init__(self,root,labels,data_set=["VOC2007"],transform=pil2tensor_transforms,jitter = 0.2,crop = False,
                batch_size = 8,train=True,
                hue=0.1, saturation=1.5, exposure=1.5):
        assert  type(labels)==list
        #data_set should be  ["VOC2007"] or ["VOC2007","VOC2012"]
        '''
        :param root:
        :param image_set:
        '''

        self.classes =labels
        self.train = train
        self.root_dir = root
        self._image_files =[]
        self._xml_files = []
        for dataset  in data_set:
            _annotationsPath = os.path.join(self.root_dir,"VOCdevkit",dataset,'Annotations',"%s.xml")
            _imagePath = os.path.join(self.root_dir,"VOCdevkit", dataset, "JPEGImages", "%s.jpg")
            set_path = os.path.join(self.root_dir,"VOCdevkit",dataset,"ImageSets","Main","%s.txt")
            if self.train:
                list_file_name = "trainval"
            else:
                list_file_name = "test"
            with open(set_path % list_file_name) as file_handler:
                lines = file_handler.readlines()
            self._image_files.extend([_imagePath %(x.rstrip()) for x in lines])
            self._xml_files.extend([_annotationsPath %(x.rstrip()) for x in lines])

        self.shape = 416
        '''
        一些图像操作参数,jiter
        配置文件的jitter=0.2，则宽高最多裁剪掉或者增加原始宽高的1/5.
        '''

        self.jitter = jitter
        self.crop = crop
        self.seen = 0
        self.batch_size = batch_size

        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

        self.transform = transform

    def get_random_size(self):
        if self.seen < 2000 * self.batch_size:
            new_size = (random.randint(0,3)+13) *32
        elif self.seen < 8000 * self.batch_size:
            new_size = (random.randint(0,3)+13) *32  #416  480
        elif self.seen < 12000 * self.batch_size:
            new_size = (random.randint(0,5)+12) * 32 #384  544
        elif self.seen <16000 * self.batch_size:
            new_size = (random.randint(0,7)+11) *32 #352  576
        else:
            new_size = (random.randint(0,9)+10) *32 #320 608
        return (new_size,new_size)

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):

        imagePath = self._image_files[index]

        labelPath = self._xml_files[index]

        if self.train:
            if self.seen %  (10 * self.batch_size)==0:
                self.shape = self.get_random_size()
            img, label = load_data_detection(imagePath,labelPath,self.classes,self.shape, self.crop, self.jitter, self.hue, self.saturation,
                                                 self.exposure)
        if self.transform is not None:
            img = self.transform(img)
        self.seen += 1

        return img,label























