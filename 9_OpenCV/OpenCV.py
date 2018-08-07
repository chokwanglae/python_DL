import numpy as np
import cv2

def showImage():
    imgfile = 'images/cat.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    cv2.namedWindow('cat', cv2.WINDOW_NORMAL)
    cv2.imshow('cat',img)
    cv2.waitKey(0)
    cv2.destoryAllWindows()


def showImage2():
    imgfile = 'images/cat.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    cv2.imshow('cat', img)

    while(True):
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == ord('c'):
            cv2.imwrite('images/cat_copy.jpg', img)
            cv2.destoryAllWindows()
        else:
            print(k)

import matplotlib.pyplot as plt
def showImage3():
    imgfile = 'images/cat.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks(range(0,250,50))
    plt.yticks([])
    plt.title('cat')
    plt.show()

showImage3()