'''
경사하강법은 미분을 통해 비용함수를 최소화하는 세타값을 찾는 방법
독립변수 X의 수가 많을 수록 유리하다.
하지만, 미분을 계속 해야하기 때문에 수많은 반복 연산이 필요하다.

X의 수가 만개가 넘으면 경사하강법을 하는 것이 유리하다고 한다.
'''
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape

# 표준정규분포로 스케일링
scaler = StandardScaler() # 객체 생성
scaled_housing_data = scaler.fit_transform(housing.data) # fit() : 변환계수 추정, transform() : 변환

n_epochs = 1000
learning_rate = 0.01
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
# mse : mean_squared_error
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("epoch:", epoch, " MSE:", mse.eval()) # mse가 줄어듦을 확인한다.
        sess.run(training_op)

    best_theta = theta.eval()
print("best_theta:")
print(best_theta)