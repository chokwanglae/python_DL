  
* 과제1 
  > 데이터 1, 데이터 2, 데이터 3을 산점도로 출력, z-score를 이용해 표준화해서 산점도 3개를 비교하기 쉽게 합쳐서 결과를 출력하라. (데이터는 임의로 한다.)

```python
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

```
  - 결과
  
<img src="https://github.com/twooopark/python_DL/blob/master/Assignment_0731/Figure_1.png" width="640px" height="480px" />
  



* 과제2
  > PCA(Principal Component Analysis)에 대해 알아오시오.
  -주성분 분석(PCA)
    - 목적 : 서로 상관관계가 있는 변수를 하나의 주성분(PC)으로 변환하는 것.
    - 변수의 값들의 변동성(분산)이 큰 순서대로 정렬을 할 수 있다.
      - 분산이 작은 순서대로 차원을 축소 해 나갈수 있다.
    - ex) 주성분(ex : PC1 = 0.5x몸무게 - 0.1x신장 + 0.4x연봉 )이 어떤 의미를 갖는지는 알기 어렵다.(여러 변수의 연산을 통해 만들어 지기 때문에, 해석이 어렵다.) 하지만, 주성분끼리 독립성을 갖기 때문에, 분석의 성능을 높일 수 있다.
    - 데이터 전처리, 모델링, 차원축소 등에 사용된다.
      - 원리(과정)
        - PC1이 가장 큰 변동의 방향을 향하고 있고(분산이 크고), PC2는 PC1에 직교한다.
        - 주성분(PC1과 PC2)를 새로운 좌표축으로 사용한다. 
        - PC2축(Y축)의 변수를 제거하여, 차원을 축소 한다.
        - 의문점 : 위 과정은, PC1의 분산은 크고, PC2의 분산은 작은 경우였다. 차원축소의 우선 순위를 정할때, 분산이 작은 순서대로 차원을 축소할까? or 분산이 큰 주성분의 직교 변수를 제거할까?
    - 장점 : 가장 뚜렷한 특징만 뽑아낸다. 차원 축소를 통한 메모리, 연산 감소
    - 단점 : 주성분이 원 변수들을 섞어 만든 것이라 해석이 어렵다. 차원 축소를 통해 디테일이 손상된다.
    
* 과제3
  > 회귀분석 정의, 
