from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    fig = plt.figure()

    # 테스트를 위해 랜덤 시드 고정
    np.random.seed(0)

    # X축 : 0~50 고정, Y축 3개 여러 랜덤값으로 생성
    X = np.arange(50)
    Y1 = np.random.random_integers(0, 100, 50)
    Y2 =  np.arange(100,300,4) + 130 * randn(50)
    Y3 = np.arange(50) + 50 * randn(50)

    # 생성한 데이터 산점도로 출력
    sp1 = fig.add_subplot(2, 3, 1)
    sp1.scatter(X, Y1, color="red" )

    sp2 = fig.add_subplot(2, 3, 2)
    sp2.scatter(X, Y2, color="blue")

    sp3 = fig.add_subplot(2, 3, 3)
    sp3.scatter(X, Y3, color="green")

    # 데이터 표준화(z-score)
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_Z = (X-X_mean)/X_std

    Y1_mean = np.mean(Y1)
    Y1_std = np.std(Y1)
    Y1_Z = (Y1-Y1_mean)/Y1_std

    Y2_mean = np.mean(Y2)
    Y2_std = np.std(Y2)
    Y2_Z = (Y2-Y2_mean)/Y2_std

    Y3_mean = np.mean(Y3)
    Y3_std = np.std(Y3)
    Y3_Z = (Y3-Y3_mean)/Y3_std

    # 표준화한 데이터 비교
    sp4 = fig.add_subplot(2, 3, 5)
    sp4.scatter(X_Z, Y1_Z, color="red" )

    sp4 = fig.add_subplot(2, 3, 5)
    sp4.scatter(X_Z, Y2_Z, color="blue")

    sp4 = fig.add_subplot(2, 3, 5)
    sp4.scatter(X_Z, Y3_Z, color="green")

    plt.show()

if __name__ == '__main__':
    ex1()
