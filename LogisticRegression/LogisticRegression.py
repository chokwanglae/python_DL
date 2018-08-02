# Logistic Regression Classifier
# 참고: http://pythonkim.tistory.com/16?category=573319 [파이쿵]
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

def ex1():
    x_data = [[1, 2],
              [2, 3],
              [3, 1],
              [4, 3],
              [5, 3],
              [6, 2]]
    y_data = [[0],
              [0],
              [0],
              [1],
              [1],
              [1]]

    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
    # 시그모이드(sigmoid) 함수는 앞에서 배운 공식(Wx + b )이 만들어 내는 값을 0과 1 사이의 값으로 변환하는 것이 목적이다.
    # 만약 WX + b가 0이면, sigmoid를 거치면 1/2 즉, 0과 1사이의 가운데인 0.5 값이 된다.
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    # cost/loss function
    # 여기서 log로 감싸는 것은 sigmoid 함수로 만들어진 hypothesis(구불구불한 cost 함수)를 매끈하게 펴기 위함이다
    # cost 함수의 목적은 비용(cost)을 판단해서 올바른 W와 b를 찾는 것이다.
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))

    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    # Launch graph
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(10001):
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
            if step % 200 == 0:
                print(step, cost_val)

        # Accuracy report
        h, c, a = sess.run([hypothesis, predicted, accuracy],
                           feed_dict={X: x_data, Y: y_data})
        print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

    '''
    0 1.73078
    200 0.571512
    400 0.507414
    600 0.471824
    800 0.447585
    ...
    9200 0.159066
    9400 0.15656
    9600 0.154132
    9800 0.151778
    10000 0.149496
    Hypothesis:  [[ 0.03074029]
     [ 0.15884677]
     [ 0.30486736]
     [ 0.78138196]
     [ 0.93957496]
     [ 0.98016882]]
    Correct (Y):  [[ 0.]
     [ 0.]
     [ 0.]
     [ 1.]
     [ 1.]
     [ 1.]]
    Accuracy:  1.0
    '''


def ex2():
    # read csv
    xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)

    # data split
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]

    print(x_data.shape, y_data.shape)

    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 8])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([8, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Hypothesis using sigmoid
    # (= tf.div(1., 1. + tf.exp(-tf.matmul(X, W))) )
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    # cost(loss) function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) +
                           (1 - Y) * tf.log(1 - hypothesis))

    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    # Launch graph
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(50001):
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
            if step % 200 == 0:
                print(step, cost_val)

        # Accuracy report
        h, c, a = sess.run([hypothesis, predicted, accuracy],
                           feed_dict={X: x_data, Y: y_data})
        print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

    '''
    0 0.82794
    200 0.755181
    400 0.726355
    600 0.705179
    800 0.686631
    ...
    9600 0.492056
    9800 0.491396
    10000 0.490767
    ...
     [ 1.]
     [ 1.]
     [ 1.]]
    Accuracy:  0.762846
    '''


ex2()