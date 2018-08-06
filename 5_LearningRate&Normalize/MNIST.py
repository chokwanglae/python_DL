import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def ex1():
    # Training Data 로드
    data_file = open('mnist_train.csv', 'r')
    # Training Data 파일의 내용을 한줄씩 불러와서 문자열 리스트로 반환
    training_data = data_file.readlines()
    # Training Data 의 두번째 데이터 확인

    def mnist(n):
        print(training_data[n][0])
        # Training Data 의 두번째 데이터를 ','로 분리
        training_data_array = np.asfarray(training_data[n].split(","))
        # 일렬로 늘어진 784 개의 픽셀 정보를 28X28 행렬로 변환
        matrix = 255 - training_data_array[1:].reshape(28,28)
        # 회색으로 Training Data 의 두번째 데이터 숫자 확인
        plt.imshow(matrix, cmap='gray')
        plt.show()

    mnist(5)

# 데이터가 많아서 나누어 학습
def ex2():
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    # more information about the mnist dataset
    # 데이터가 없을 경우 자동으로 다운받음(시간이 어느정도 걸림)
    # one_hot = True로 하면 lable 데이터가 one_hot 방식으로 나옴
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 분류 숫자 0 ~ 9
    nb_classes = 10

    # MNIST data image of shape 28 * 28 = 784
    X = tf.placeholder(tf.float32, [None, 784])

    # 0 - 9 digits recognition = 10 classes
    Y = tf.placeholder(tf.float32, [None, nb_classes])

    W = tf.Variable(tf.random_normal([784, nb_classes]))
    b = tf.Variable(tf.random_normal([nb_classes]))

    # Hypothesis (using softmax)
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

    # cross_entropy
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    # Test model
    is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # parameters
    # epoch : 전체 데이터를 학습하는 횟수
    # batch : 한번에 메모리에 올리는 데이터 수
    training_epochs = 15
    batch_size = 100

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            # 전체 데이터의 개수를 batch_size로 나누면 1epoch에 필요한 횟수를 구할 수 있다.
            # 전체사이즈(10000)/100 =100번 돌면 1번
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                # Training data를 이용하여 학습
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 100개씩 돌면서 학습
                c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
                avg_cost += c / total_batch
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
            # Epoch: 0015 cost = 0.466380322

        # Test data를 활용하여 정확도 측정
        # sess.run()과 tensor.eval은 같은 기능이다.
        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
        # Accuracy: 0.8886

        # Test data 중 하나를 임의로 뽑아서 테스트
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
        # Label: [3]
        # Prediction: [3]

        # 화면에 출력
        plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()

ex2()