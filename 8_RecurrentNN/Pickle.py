'''
pickle 모듈
일반 텍스트를 파일로 저장할 때는 파일 입출력을 이용한다.
하지만 리스트나 클래스같은 텍스트가 아닌 자료형은 일반적인 파일 입출력 방법으로는 데이터를 저장하거나 불러올 수 없다.
파이썬에서는 이와 같은 텍스트 이외의 자료형을 파일로 저장하기 위하여 pickle이라는 모듈을 제공한다.
'''

import pickle
data = {
    'a' : [1, 2.0, 3, 4+6j],
    'b' : ("character string", b"byte string"),
    'c' : {None, True, False}
}

def write():
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)

def read():
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
        print(data)