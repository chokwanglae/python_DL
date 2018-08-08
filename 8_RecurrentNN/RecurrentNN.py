'''
RNN은 다양한 자연어처리(NLP) 문제에 대해 뛰어난 성능을 보이고 있는 인기있는 모델
Hidden Node가 방향을 가진 엣지로 연결돼 순환구조를 이루는(Directed Cycle) 인공신경망의 한 종류
RNN에 대한 기본적인 아이디어는 순차적인 정보를 처리
기존의 신경망 구조에서는 모든 입력(과 출력)이 각각 독립적이라고 가정

RNN의 여러 종류 중에서 가장 많이 사용되는 것은 LSTM으로,
기본 RNN 구조에 비해 더 긴 시퀀스를 효과적으로 잘 기억하기 때문이다.
LSTM은 우선은 대략적으로 hidden state를 계산하는 방법만 조금 다를뿐, RNN과 기본적으로는같다.

자동번역(기계번역), 음성 인식, 이미지 캡션 생성(CNN+RNN)
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

# 예제Input 4 dim->2 hidden_size
def ex1():
    pp = pprint.PrettyPrinter(indent=4)
    sess = tf.InteractiveSession()

    with tf.variable_scope('one_cell') as scope:
        # One cell RNN input_dim (4) -> output_dim (2)
        hidden_size = 2
        cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
        print(cell.output_size, cell.state_size)

        x_data = np.array([[[1,0,0,0]]], dtype=np.float32)
        pp.pprint(x_data)
        outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

        sess.run(tf.global_variables_initializer())
        pp.pprint(outputs.eval())

# HELLO 예제 ( 입력과 출력의 shape 변화 확인)
def ex2():
    pp = pprint.PrettyPrinter(indent=4)
    sess = tf.InteractiveSession()

    with tf.variable_scope('two_sequences') as scope:
        # One cell RNN input_dim (4) -> output_dim (2), sequence : 5
        hidden_size = 2
        cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)

        x_data = np.array([
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]]
        ], dtype=np.float32)
        print(x_data.shape)
        pp.pprint(x_data) #

        outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
        sess.run(tf.global_variables_initializer())
        pp.pprint(outputs.eval())

def ex3():
    idx2char = ['h', 'i', 'e', 'l', 'o']

    # Teach hello : hihell -> ihello
    x_data = [[0, 1, 0 ,2, 3, 3]] # hihell : 인덱스
    x_one_hot = [[[1, 0, 0, 0, 0], # 학습시키고자 하는 문자열
                  [0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0],
                  ]]
    y_data = [[1, 0, 2, 3, 3, 4]] # ihello : 출력하고자 하는 라벨

    num_classes = 5
    input_dim = 5 # one-hot size
    hidden_size = 5 # output from the LSTM. 5 to directly predict one-hot
    batch_size = 1 # one sentence
    sequence_length = 6 # ihello.length == 6
    learning_rate = 0.1

    X = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) # X one-hot
    Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32) # 출력 5
    outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

    weights = tf.ones([batch_size, sequence_length])
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    prediction = tf.argmax(outputs, axis=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
            result = sess.run(prediction, feed_dict={X: x_one_hot})
            print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

            # print char using dic
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print("\tPrediction str: ", ''.join(result_str))

# LongSequenceRNN
def ex4():
    sample = " if you want you"
    idx2char = list(set(sample)) # index -> char
    char2idx = {c: i for i, c in enumerate(idx2char)} # char -> idx

    # hyper parameters
    dic_size = len(char2idx) # RNN input size (one hot size)
    hidden_size = len(char2idx) # RNN ouput size
    num_classes = len(char2idx) # final output size(RNN or softmax, etc.)
    batch_size = 1 # one sample data, one batch
    sequence_length = len(sample) - 1 # number of lstm rollings (unit #)
    learning_rate = 0.1

    sample_idx = [char2idx[c] for c in sample] # char to idx
    x_data = [sample_idx[:-1]]  # X data sample (0~ n-1) hello: hell
    y_data = [sample_idx[1:]] # Y label sample (1~ n) hello: ello

    X = tf.placeholder(tf.int32, [None, sequence_length]) # X_data
    Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y Label

    X_one_hot = tf.one_hot(X, num_classes) # one hot : 1 -> 0 1 0 0 0 0 0 0 0 0

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

    weights = tf.ones([batch_size, sequence_length])
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    prediction = tf.argmax(outputs, axis=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
            result = sess.run(prediction, feed_dict={X: x_data})

            # print char using dic
            result_str = [idx2char[c] for c in np.squeeze(result)]

            print(i, "loss: ", l, " Prediction: ", ''.join(result_str))

# Really long sentence RNN
def ex5():
    sample = ("if you want to build a ship, don't drum up people together to collect"
    "wood and don't assign them tasks and work, but rather teach them to long for"
    "the endless immensity of the sea.")


def ex6():
    seed = 0
    np.random.seed(seed)
    # 원소가 9개인 numpy 배열을 생성한다.
    # Y값은 0이 3개, 1은 6개로 비율은 1:2 이다 (불균형데이터)
    X = np.array([-5, -3, -1, 1, 3, 5, 7, 9, 11])
    Y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    splits = 3

ex4()