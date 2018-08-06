# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Placeholder : 실제로 아무 계산도 하지 않는 특수한 노드, 실행 시 데이터를 출력한다.
훈련하는 동안, 텐서에 훈련 데이터를 전달하기 위해 사용한다.
(실행 시 placeholder에 값을 지정하지 않으면 예외 발생)
'''
import tensorflow as tf

def ex1():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)

    adder_node = a+b

    sess = tf.Session()
    print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
    print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))

    add_and_triple = adder_node * 3

    print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))

def ex2():
    A = tf.placeholder(tf.float32, shape=(None, 3))
    B = A+5
    with tf.Session() as sess:
        B_val1 = B.eval(feed_dict={A: [[1, 2, 3]]})
        B_val2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
    print(B_val1)
    print(B_val2)
ex2()