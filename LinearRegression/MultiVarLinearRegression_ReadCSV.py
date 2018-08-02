# Multi-variable linear regression
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

def ex3():
    xy = np.loadtxt('test-score.csv', delimiter=',', dtype=np.float32)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]

    # 데이터 잘 읽어들였는지 확인
    print(x_data.shape, x_data, len(x_data))
    print(y_data.shape, y_data)

    # placeholders for a tensor that will be always fed.
    # None으로 설정하면, 행의 갯수를 지정하지 않고 받는다.
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # matmul: 행렬곱, Hypothesis
    hypothesis = tf.matmul(X, W) + b

    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # sess.run()을 통해 읽어들인 데이터로 학습한다.
    for step in range(2001):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

    # Ask my score
    # 학습모델로 점수를 예측해본다.
    print("Your score will be ", sess.run(
        hypothesis, feed_dict={X: [[100, 70, 101]]}))

    print("Other scores will be ", sess.run(hypothesis,
                                            feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

    '''
    Your score will be  [[ 181.73277283]]
    Other scores will be  [[ 145.86265564]
     [ 187.23129272]]
    '''

def ex4():

    # tf.train.string_input_producer()으로 필요 파일들을
    # 랜덤(T,F 설정 가능)으로 filename queue 에 추가(enqueue)
    filename_queue = tf.train.string_input_producer(
        string_tensor=['test-score.csv'], shuffle=False, name='filename_queue')

    # 데이타 포맷 (csv, tfrecord, TextLineReader 등) 에 맞는 reader 를 통해서
    # filename queue 에서 dequeue 된 파일들을 value에 담는다.
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    # value들을 csv로 decode 한다.
    # http://bcho.tistory.com/1165?category=555440
    record_defaults = [[0.], [0.], [0.], [0.]] # (csv 형식)
    xy = tf.decode_csv(value, record_defaults=record_defaults)


    # collect batches of csv in
    train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=25)


    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Hypothesis
    hypothesis = tf.matmul(X, W) + b

    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    for step in range(2001):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

    coord.request_stop()
    coord.join(threads)

    # Ask my score
    print("Your score will be ",
          sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

    print("Other scores will be ",
          sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

    '''
    Your score will be  [[ 177.78144836]]
    Other scores will be  [[ 141.10997009]
     [ 191.17378235]]
    '''
ex4()
