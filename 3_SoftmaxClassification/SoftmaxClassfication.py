import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

def ex1():
    x_data = [[1, 2, 1, 1],
              [2, 1, 3, 2],
              [3, 1, 3, 4],
              [4, 1, 5, 5],
              [1, 7, 5, 5],
              [1, 2, 5, 6],
              [1, 6, 6, 6],
              [1, 7, 7, 7]]
    y_data = [[0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [1, 0, 0]]

    X = tf.placeholder("float", [None, 4]) # N행 4열
    Y = tf.placeholder("float", [None, 3]) # N행 3열
    nb_classes = 3 # nb :

    W = tf.Variable(tf.random_normal([4, nb_classes]), name = 'weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

    '''
    참고(logistic)
        # Hypothesis using sigmoid
        # (= tf.div(1., 1. + tf.exp(-tf.matmul(X, W))) )
        hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
    
        # cost(loss) function
        cost = -tf.reduce_mean(Y * tf.log(hypothesis) +
                               (1 - Y) * tf.log(1 - hypothesis))
    '''


    # tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

    # Cross entropy cost(loss)
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
            if step % 200 == 0:
                print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

            # Testing & One-hot encoding
            # One-hot encoding : N개 중에 단 한개만 의미있도록 하는 것.
            a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
            # argmax 가장 높은 값( 첫번째 access 기준)
            print(a, sess.run(tf.argmax(a, 1)))

def ex2():

    # Predicting animal type based on various features
    xy = np.loadtxt('zoo.csv', delimiter=',', dtype=np.float32)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]

    print(x_data.shape, y_data.shape)

    nb_classes = 7  # 0 ~ 6

    X = tf.placeholder(tf.float32, [None, 16])
    Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
    Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
    print("one_hot", Y_one_hot)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
    print("reshape", Y_one_hot)

    W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

    # tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    logits = tf.matmul(X, W) + b
    hypothesis = tf.nn.softmax(logits)

    # Cross entropy cost/loss
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=Y_one_hot)
    cost = tf.reduce_mean(cost_i)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    # argmax : 가장 높은 값( 첫번째 access 기준)
    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Launch graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2000):
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
            if step % 100 == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={
                    X: x_data, Y: y_data})
                print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                    step, loss, acc))

        # Let's see if we can predict
        pred = sess.run(prediction, feed_dict={X: x_data})
        # y_data: (N,1) = flatten => (N, ) matches pred.shape
        for p, y in zip(pred, y_data.flatten()):
            print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

    '''
    Step:     0 Loss: 5.106 Acc: 37.62%
    Step:   100 Loss: 0.800 Acc: 79.21%
    Step:   200 Loss: 0.486 Acc: 88.12%
    Step:   300 Loss: 0.349 Acc: 90.10%
    Step:   400 Loss: 0.272 Acc: 94.06%
    Step:   500 Loss: 0.222 Acc: 95.05%
    Step:   600 Loss: 0.187 Acc: 97.03%
    Step:   700 Loss: 0.161 Acc: 97.03%
    Step:   800 Loss: 0.140 Acc: 97.03%
    Step:   900 Loss: 0.124 Acc: 97.03%
    Step:  1000 Loss: 0.111 Acc: 97.03%
    Step:  1100 Loss: 0.101 Acc: 99.01%
    Step:  1200 Loss: 0.092 Acc: 100.00%
    Step:  1300 Loss: 0.084 Acc: 100.00%
    ...
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 0 True Y: 0
    '''

ex1()