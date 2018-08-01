# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

def ex1():
    x1_data = [73., 93., 89., 96., 73.]
    x2_data = [80., 88., 91., 98., 66.]
    x3_data = [75., 93., 90., 100., 70.]

    y_data = [152., 185., 180., 196., 142.]

    # placeholders for a tensor that will be always fed.
    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)

    Y = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.random_normal([1]), name='weight1')
    w2 = tf.Variable(tf.random_normal([1]), name='weight2')
    w3 = tf.Variable(tf.random_normal([1]), name='weight3')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
    print(hypothesis)

    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Minimize. Need a very small learning rate for this data set
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

    '''
    0 Cost:  19614.8
    Prediction:
     [ 21.69748688  39.10213089  31.82624626  35.14236832  32.55316544]
    10 Cost:  14.0682
    Prediction:
     [ 145.56100464  187.94958496  178.50236511  194.86721802  146.08096313]
     ...
    1990 Cost:  4.9197
    Prediction:
     [ 148.15084839  186.88632202  179.6293335   195.81796265  144.46044922]
    2000 Cost:  4.89449
    Prediction:
     [ 148.15931702  186.8805542   179.63194275  195.81971741  144.45298767]
    '''
def ex2():
    x_data = [[73., 80., 75.],
              [93., 88., 93.],
              [89., 91., 90.],
              [96., 98., 100.],
              [73., 66., 70.]]
    y_data = [[152.],
              [185.],
              [180.],
              [196.],
              [142.]]

    # placeholders for a tensor that will be always fed.
    # None으로 설정하면, 행의 갯수를 지정하지 않고 받는다.
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

    for step in range(2001):
        # hy와 cost 그래프를 계산, 가설 수식에 넣어야 할 실제 값을 feed_dict로 전달
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

    '''
    0 Cost:  7105.46
    Prediction:
     [[ 80.82241058]
     [ 92.26364136]
     [ 93.70250702]
     [ 98.09217834]
     [ 72.51759338]]
    10 Cost:  5.89726
    Prediction:
     [[ 155.35159302]
     [ 181.85691833]
     [ 181.97254944]
     [ 194.21760559]
     [ 140.85707092]]
    ...
    1990 Cost:  3.18588
    Prediction:
     [[ 154.36352539]
     [ 182.94833374]
     [ 181.85189819]
     [ 194.35585022]
     [ 142.03240967]]
    2000 Cost:  3.1781
    Prediction:
     [[ 154.35881042]
     [ 182.95147705]
     [ 181.85035706]
     [ 194.35533142]
     [ 142.036026  ]]
    '''

ex2()