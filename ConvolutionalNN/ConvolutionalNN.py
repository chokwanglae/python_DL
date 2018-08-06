import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 기초d
def ex0():
    sess = tf.InteractiveSession()
    image = np.array([[[[1], [2], [3]],
                       [[4], [5], [6]],
                       [[7], [8], [9]]]], dtype=np.float32)
    print(image.shape)
    plt.imshow(image.reshape(3, 3), cmap='Greys')

    plt.show()


# filter, padding 적용 전
def ex1():
    image = np.array([
        [[[1], [2], [3]],
         [[4], [5], [6]],
         [[7], [8], [9]]]
    ], dtype = np.float32)

    print("image.shape", image.shape)
    weight = tf.constant([
        [[[1.]],[[1.]]],
        [[[1.]],[[1.]]],
    ])
    print("weight.shape", weight.shape)
    sess = tf.InteractiveSession()
    conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
    conv2d_img = conv2d.eval()

    print("conv2d_img.shape", conv2d_img.shape)
    conv2d_img = np.swapaxes(conv2d_img, 0, 3)

    for i, one_img in enumerate(conv2d_img):
        print(one_img.reshape(2, 2))
        plt.subplot(1, 2, i+1), plt.imshow(one_img.reshape(2, 2))
    plt.show()

# filter, padding 적용
def ex2():
    image = np.array([
        [[[1], [2], [3]],
         [[4], [5], [6]],
         [[7], [8], [9]]]
    ], dtype = np.float32)

    print("image.shape", image.shape)
    weight = tf.constant([
        [[[1., 10., -1.]],[[1., 10., -1.]]], # [[[1.]],[[1.]]], filter
        [[[1., 10., -1.]],[[1., 10., -1.]]],
    ])
    print("weight.shape", weight.shape)
    sess = tf.InteractiveSession()
    conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME') #'VALID')
    conv2d_img = conv2d.eval()

    print("conv2d_img.shape", conv2d_img.shape)
    conv2d_img = np.swapaxes(conv2d_img, 0, 3)

    for i, one_img in enumerate(conv2d_img):
        print(one_img.reshape(3, 3))
        plt.subplot(1, 3, i+1), plt.imshow(one_img.reshape(3, 3))
    plt.show()

ex2()