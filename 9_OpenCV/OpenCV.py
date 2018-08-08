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

def showVideo1():
    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture(0)
    except:
        print('카메라 구동 실패')
        return

    cap.set(3, 680)
    cap.set(4, 320)

    while True:
        ret, frame = cap.read()

        if not ret:
            print('비디오 읽기 오류')
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('video'.gray)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def drawing():
    img = np.zeros((512, 512, 3), np.uint8)

    # 다양한 색상과 선두께를 가진 도형 그리기
    cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    cv2.circle(img, (477, 63), 63, (0, 0, 255), -1)
    cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, (255, 0, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'sess', (10, 500), font, 6, (255, 255, 255), 5)

    cv2.imshow('drawing', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

from random import shuffle
import math

mode, drawing = True, False
ix, iy = -1, -1
R = [i for i in range(256)]
G = [i for i in range(256)]
B = [i for i in range(256)]



def mouseBrush():
    global mode

    def onMouse(event, x, y, flags, param):
        global ix, iy, drawing, mode, R, G, B
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            shuffle(R), shuffle(G), shuffle(B)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if mode:
                    cv2.rectangle(param, (ix, iy), (x, y), (R[0], G[0], B[0]), -1)
                else:
                    r = (ix - x) ** 2 + (iy - y) ** 2
                    r = int(math.sqrt(r))
                    cv2.circle(param, (ix, iy), r, (R[0], G[0], B[0]), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode:
                cv2.rectangle(param, (ix, iy), (x, y), (R[0], G[0], B[0]), -1)
            else:
                r = (ix - x) ** 2 + (iy - y) ** 2
                r = int(math.sqrt(r))
                cv2.circle(param, (ix, iy), r, (R[0], G[0], B[0]), -1)

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('paint')
    cv2.setMouseCallback('paint', onMouse, param=img)

    while True:
        cv2.imshow('paint', img)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break
        elif k == ord('m'):
            mode = not mode

    cv2.destroyAllWindows()

# mouseBrush()



def addImage(imgfile1, imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    add_img1 = img1 + img2
    add_img2 = cv2.add(img1, img2)

    cv2.imshow('img1+img2', add_img1)
    cv2.imshow('add(img1, img2)', add_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# addImage('images/hallstatt.jpg', 'images/suji.jpg')


def imgBlending(imgfile1, imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    def onMouse(x):
        pass
    cv2.namedWindow('ImgPane')
    cv2.createTrackbar('MIXING', 'ImgPane', 0, 100, onMouse)
    mix=cv2.getTrackbarPos('MIXING', 'ImgPane')

    while True:
        img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100, 0)
        cv2.imshow('ImgPane', img)
        k = cv2.waitKey(1) & 0xFF
        mix = cv2.getTrackbarPos('MIXING', 'ImgPane')

    cv2.destroyAllWindows()

# imgBlending('images/hallstatt.jpg', 'images/suji.jpg')


def bitOperation(hpos, vpos):
    img1 = cv2.imread('images/suji.jpg')
    img2 = cv2.imread('images/logo.jpg')

    rows, cols, channels = img2.shape
    roi = img1[vpos:rows+vpos, hpos:cols+hpos]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_bg = cv2.bitwise_and(img2, img2, mask=mask)

    dst = cv2.add(img1_bg, img2_bg)
    img1[vpos:rows+vpos, hpos:cols+hpos] = dst

    cv2.imshow('result ', img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

bitOperation(10, 10)