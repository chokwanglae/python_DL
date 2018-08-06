from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

# basic
def ex1():
    img = mnist.train.images[0].reshape(28, 28)
    sess = tf.InteractiveSession()
    img = img.reshape(-1, 28, 28, 1)
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
    conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')  # 2칸씩 이미지 이동(7 X 7)
    print(conv2d)

    sess.run(tf.global_variables_initializer())
    conv2d_img = conv2d.eval()
    conv2d_img = np.swapaxes(conv2d_img, 0, 3)
    for i, one_img in enumerate(conv2d_img):
        plt.subplot(1, 5, i + 1), plt.imshow(one_img.reshape(14, 14), cmap='gray')
    plt.show()

# maxpooling
def ex2():
    img = mnist.train.images[0].reshape(28, 28)
    sess = tf.InteractiveSession()
    img = img.reshape(-1, 28, 28, 1)
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
    conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')  # 2칸씩 이미지 이동(7 X 7)
    pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # maxpooling 추가
    print(pool)

    sess.run(tf.global_variables_initializer())
    pool_img = pool.eval()
    pool_img = np.swapaxes(pool_img, 0, 3)
    for i, one_img in enumerate(pool_img):
        plt.subplot(1, 5, i + 1), plt.imshow(one_img.reshape(7, 7), cmap='gray')
    plt.show()


ex2()

def ex3():
    # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
    keep_prob = tf.placeholder(tf.float32)

    # input place holders
    X = tf.placeholder(tf.float32, [None, 784])
    X_img = tf.reshape(X, [-1, 28, 28, 1])  # img 28x28x1 (black/white)
    Y = tf.placeholder(tf.float32, [None, 10])

    # L1 ImgIn shape=(?, 28, 28, 1)
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    #    Conv     -> (?, 28, 28, 32)
    #    Pool     -> (?, 14, 14, 32)
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    '''
    Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
    Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
    Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
    Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
    '''

    # L2 ImgIn shape=(?, 14, 14, 32)
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    #    Conv      ->(?, 14, 14, 64)
    #    Pool      ->(?, 7, 7, 64)
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    '''
    Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
    Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
    Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
    Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
    '''

    # L3 ImgIn shape=(?, 7, 7, 64)
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    #    Conv      ->(?, 7, 7, 128)
    #    Pool      ->(?, 4, 4, 128)
    #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
    '''
    Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
    Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
    Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
    Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
    Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
    '''