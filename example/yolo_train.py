import logging
import time

import sys
sys.path.insert(0, '.')

from yolonet.utils import  initEnv
logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
         level=logging.DEBUG)
log = logging.getLogger(__name__)

from yolonet.network.engine import  VOCTrianningEngine


def main():
    config = initEnv(1, "Yolov3")
    VOCTrianningEngine(config)

if __name__ == '__main__':
    main()
