'''
정규방정식은 미분을 수행하지 않고, 빠르게 세타값을 찾을 수 있다.
독립변수 X의 수가 많을수록 불리하다.

X의 수가 만개가 넘으면 경사하강법을 하는 것이 유리하다고 한다.
'''

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape

# 편향에 대한 입력 특성(X0 = 1)을 추가
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

# 전치 행렬
XT = tf.transpose(X)

# inverse 역행렬
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

print(X)
print(y)
print(XT)
print(theta)
with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)


