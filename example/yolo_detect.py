import logging
import time
#from PIL import Image
import sys
sys.path.insert(0, '.')

from yolonet.utils import  initEnv
logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
         level=logging.DEBUG)
log = logging.getLogger(__name__)

from yolonet.network.engine import  YoloV3DetectEngine


def main():
    config = initEnv(2, "Yolov3")
    eng=YoloV3DetectEngine(config)
    #results, img = eng.detectionOneImage("data/ten_18596.jpg")

    #img.show()
    img_b = eng.evaluation()
    #img_a.save("1.jpg")
    #img_b.save("2.jpg")
if __name__ == '__main__':
    main()
