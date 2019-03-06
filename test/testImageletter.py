
import cv2
import PIL.Image as Image
import  PIL

from yolonet.data.transforms import LetterBoxImage


op = LetterBoxImage(608)



def testcv2():
    img = cv2.imread("data/ten_18583.jpg")
    result = op(img)
    cv2.imwrite("data/ten_cv2.jpg",result)

def testPil():
    img = Image.open("data/ten_18583.jpg")
    result = op(img)
    result.save("data/ten_pil.jpg","jpeg")

if __name__ == '__main__':
    testcv2()
    testPil()
