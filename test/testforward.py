#!/usr/bin/python3


from  PIL import Image

from  yolonet.network.module._yolov3 import  Yolov3
import torch
from  yolonet.data.transforms._preprocess import LetterBoxImage
from yolonet.utils import initEnv
import torchvision.transforms as transforms
from yolonet.data.transforms._postprocess import CorrectDetection,DrawRectengle
env = initEnv(2,"Yolov3")
letter_tf = LetterBoxImage(env["input_shape"])

import pickle as pkl
import  time
class AddDim(object):
    def __call__(self,tensor):
        '''
        单张图片的时候增加一维度
        :param tensor:
        :return:
        '''
        shape = list(tensor.shape)
        shape.insert(0, 1)
        image_ = tensor.view(tuple(shape))
        return image_



def loadImage(imagefile):
    image = Image.open(imagefile)
    img = letter_tf(image)
    return image,img


def main():
    ad = AddDim()
    tt = transforms.ToTensor()
    ori_img,img=loadImage(env)
    img = ad(tt(img))
    config = initEnv(2, "Yolov3")
    correctDetection = CorrectDetection(config["net_size"])
    model = Yolov3(config,test_flag=2)
    model.load_weights("data/190221_final.weights")
    start_time = time.time()
    model.eval()
    ouput = model(img)
    end_time = time.time()
    print("c lib nms spend time {}".format(end_time-start_time))
    print(ouput)

    #co_tf = CorrectDetection(config["input_shape"])
    #detections = co_tf(ouput,ori_img.size)


def test2img():
    ad = AddDim()
    tt = transforms.ToTensor()
    ori_img_a,img_a=loadImage("data/ten_18583.jpg")
    ori_img_b,img_b = loadImage("data/ten_18596.jpg")

    img_a = ad(tt(img_a))
    img_b = ad(tt(img_b))
    batch_data = torch.cat((img_a,img_b),0)
    config = initEnv(2, "Yolov3")
    model = Yolov3(config,test_flag=2)
    correctDetection = CorrectDetection(config["input_shape"])
    model.load_weights("data/190221_final.weights")
    start_time = time.time()
    model.eval()
    ouput = model(batch_data)
    end_time = time.time()
    print("c lib nms spend time {}".format(end_time-start_time))
    print(len(ouput))
    print(ouput)
    for i  in ouput:
        correctDetection(i,(1920,1080))
    img_a = DrawRectengle(ori_img_a,ouput[0],config["labels"])
    img_b = DrawRectengle(ori_img_b,ouput[1],config["labels"])
    img_a.save("1.jpg","JPEG")
    img_b.save('2.jpg',"JPEG")








if __name__ == '__main__':
    test2img()



