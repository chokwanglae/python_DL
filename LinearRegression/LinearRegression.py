# Linear Regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# sess.run(train)
def ex1():
    # X and Y data
    x_train = [1, 2, 3]
    y_train = [1, 2, 3]

    # Try to find values for W and b to compute y_data = x_data * W + b
    # We know that W should be 1 and b should be 0
    # But let TensorFlow figure it out
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Our hypothesis XW+b
    hypothesis = x_train * W + b

    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)
    '''
    print(train) 결과
    name: "GradientDescent"
    op: "NoOp"
    input: "^GradientDescent/update_weight/ApplyGradientDescent"
    input: "^GradientDescent/update_bias/ApplyGradientDescent"
    '''

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(cost), sess.run(W), sess.run(b))

    # Learns best fit W:[ 1.],  b:[ 0.]


# 학습 데이터가 담긴 feed dictionary를 이용, 입실론 b 추가
def ex2():
    # X and Y data
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # Try to find values for W and b to compute y_data = x_data * W + b
    # We know that W should be 1 and b should be 0
    # But let TensorFlow figure it out
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Our hypothesis XW+b
    hypothesis = X * W + b

    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the line with new training data
    for step in range(2001):
        cost_val, W_val, b_val, _ = \
            sess.run([cost, W, b, train], # fetch할 값
                     feed_dict={X: [1, 2, 3, 4, 5], # 학습 데이터
                                Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
    # Learns best fit W:[ 1.],  b:[ 1.1.]


    # W, b 대신 hypothesis 출력

    # for step in range(2001):
    #     cost_val, hy_val, _ = \
    #         sess.run([cost, hypothesis, train], # fetch할 값
    #                  feed_dict={X: [1, 2, 3, 4, 5], # 학습 데이터
    #                             Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    #     if step % 20 == 0:
    #         print(step, cost_val, hy_val)


ex2()