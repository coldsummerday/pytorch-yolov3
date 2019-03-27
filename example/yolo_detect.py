import logging
import sys
sys.path.insert(0, '.')

from yolonet.utils import  initEnv
logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
         level=logging.DEBUG)
log = logging.getLogger(__name__)

from yolonet.network.engine import  YoloV3DetectEngine
import  argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Yolo v3 for pytorch')
    parser.add_argument('model_name', help='model name', type=str, default="Yolov3")
    parser.add_argument("image",help="detect image name")
    args = parser.parse_args()
    return args



def main():
    arg = parse_args()
    config = initEnv(2, arg.model_name)
    eng=YoloV3DetectEngine(config)
    results, img = eng.detectionOneImage(arg.image)

    img.show()

if __name__ == '__main__':
    main()
