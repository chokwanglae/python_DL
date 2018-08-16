'''
1. DT, SVM, SVDD, Logistic 정의
2. 위 알고리즘을 사용해 타이타닉 데이터 분석
2. 결과물 한 눈에 볼 수 있도록 (정확도)
3. 유인물 모르는 단어, 백과사전을 찾아서라도 정리
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 읽기
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 데이터 파악
'''
# 총 891명, 생존률 38.3%
       PassengerId    Survived     ...           Parch        Fare
count   891.000000  891.000000     ...      891.000000  891.000000
mean    446.000000    0.383838     ...        0.381594   32.204208
std     257.353842    0.486592     ...        0.806057   49.693429
'''
print(train.describe())

print(train.head())
print(test.head()) # Survived 컬럼이 없음 (label이 없음)

# 결측값 현황 파악

print(train.isnull().sum()) # Age: 177, Cabin : 687, Embarked : 2
print(test.isnull().sum()) # Age : 86, Fare : 1, Cabin : 327

# 선실 등급별 파악
'''
# 선실 등급이 높을 수록, 생존률이 높음
        PassengerId  Survived    ...         Parch       Fare
Pclass                           ...                         
1        461.597222  0.629630    ...      0.356481  84.154687
2        445.956522  0.472826    ...      0.380435  20.662183
3        439.154786  0.242363    ...      0.393075  13.675550
'''
print(train.groupby('Pclass').mean())

# 상관관계 분석
'''
수치가 0 에 가까울수록, 상관관계가 낮다
Pclass : 0.34
Fare : 0.26
이 두 독립변수가 Survived와 상관관계가 높다.

Pclass - Fare : 0.55의 가장 큰 상관관계를 가짐(종속성을 나타내므로 제거해보자)
SibSp - Parch : 0.41 가족 관계이므로 당연히 큼
SibSp와 Parch - Fare : 0.16, 0.22로 티켓을 같이 사서 높게 책정됐다고 생각됨

'''
plt.figure(figsize=(10, 10))
sns.heatmap(train.corr(), linewidths=0.01, square=True,
            annot=True, cmap=plt.cm.viridis, linecolor="white")
plt.title('Correlation between features')
plt.show()

# 전처리를 위해 데이터 합침. (List)
train_test_data = [train, test]

exit(0)
# Name 컬럼 전처리

# 1. 이름에서 '.' 앞에있는 단어 하나(title)를 추출한다.
'''
(titles) 
(of a man): Mr (Mister, mister), Sir (sir); 
(of a woman): Ms (Miz, mizz), Mrs (Mistress, mistress), Miss (miss), Dame (dame), 
(of a non-binary person): Mx (Mixter); 
(see also): Dr (Doctor, doctor), Madam (madam, ma'am)
'''
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()
test['Title'].value_counts()

# Mr = mr, mister,
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 0, "Rev": 0, "Col": 0, "Major": 0, "Mlle": 0,"Countess": 1,
                 "Ms": 1, "Lady": 1, "Jonkheer": 0, "Don": 0, "Dona" : 0, "Mme": 0,"Capt": 0,"Sir": 0 }

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)



# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# male: 0, female: 1
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)



# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

'''
Binning/Converting Numerical Age to Categorical Variable  

feature vector map:  
child: 0  
young: 1  
adult: 2  
mid-age: 3  
senior: 4
'''
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 18, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 35), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 43), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 43) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

print(train.head())

'''
more than 50% of 1st class are from S embark  
more than 50% of 2nd class are from S embark  
more than 50% of 3rd class are from S embark

**fill out missing embark with S embark**
'''
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print(train.head())

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

print(train.head())

for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 10.5, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[ dataset['Fare'] > 30, 'Fare'] = 2

print(train.head())


train.Cabin.value_counts()
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

cabin_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

print(train.head())


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

print(train.head())

features_drop = ['Ticket', 'SibSp', 'Parch', 'Cabin']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape

print(train_data.head())

# Importing Classifier Modules
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# kNN Score
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("KNN: ",round(np.mean(score)*100, 2))

# decision tree Score
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("DT: ",round(np.mean(score)*100, 2))


# Ramdom Forest
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("RF: ",round(np.mean(score)*100, 2))



# Naive Bayes Score
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("NB: ",round(np.mean(score)*100, 2))


# SVM
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print("SVC: ",round(np.mean(score)*100,2))


# CV + Random Forest
def RF_Best():
    # estimator_grid = np.arange(10, 22, 1)
    # depth_grid = np.arange(3, 6, 1)
    # parameters = {'n_estimators': estimator_grid, 'max_depth': depth_grid}
    # gridCV = GridSearchCV(RandomForestClassifier(), param_grid=parameters, cv=10)
    # gridCV.fit(train_data, target)
    best_n_estim = 20#gridCV.best_params_['n_estimators']
    best_depth = 5#gridCV.best_params_['max_depth']
    print("best: ",best_depth, best_n_estim)

    RF_best = RandomForestClassifier(max_depth=best_depth,n_estimators=best_n_estim,random_state=3)
    # RF_best.fit(train_data, target)
    score = cross_val_score(RF_best, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    print(score)
    print("RF_best: ",round(np.mean(score)*100, 2))

    clf = RandomForestClassifier(max_depth=best_depth, n_estimators=best_n_estim, random_state=3)
    clf.fit(train_data, target)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('submission.csv', index=False)
    submission = pd.read_csv('submission.csv')
    submission.head()


# CV + AdaBoost
def AB_Best():
    estimator_grid = np.arange(65, 75, 1)
    learning_rate_grid = np.array([0.2,0.23,0.26,0.29])
    parameters = {'n_estimators': estimator_grid, 'learning_rate': learning_rate_grid}
    gridCV = GridSearchCV(AdaBoostClassifier(), param_grid=parameters, cv=10)
    gridCV.fit(train_data, target)
    best_n_estim = gridCV.best_params_['n_estimators']
    best_learn_rate = gridCV.best_params_['learning_rate']
    print("Ada Boost best n estimator : " + str(best_n_estim))
    print("Ada Boost best learning rate : " + str(best_learn_rate))

    AB_best = AdaBoostClassifier(n_estimators=best_n_estim,learning_rate=best_learn_rate,random_state=3)
    AB_best.fit(train_data, target);
    score = cross_val_score(AB_best, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    print(score)
    print("AdaBoost: ",round(np.mean(score)*100, 2))

    clf = AdaBoostClassifier(n_estimators=best_n_estim, learning_rate=best_learn_rate, random_state=3)
    clf.fit(train_data, target)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('submission.csv', index=False)
    submission = pd.read_csv('submission.csv')
    submission.head()


# CV + SVM
def SVM_Best():
    C_grid = [0.1, 0.3, 0.6]
    gamma_grid = [0.3, 0.6, 0.9]
    parameters = {'C': C_grid, 'gamma': gamma_grid}
    gridCV = GridSearchCV(SVC(kernel='rbf'), parameters, cv=10);
    gridCV.fit(train_data, target)
    best_C = gridCV.best_params_['C']
    best_gamma = gridCV.best_params_['gamma']

    print("SVM best C : " + str(best_C))
    print("SVM best gamma : " + str(best_gamma))

    SVM_best = SVC(C=best_C, gamma=best_gamma)
    SVM_best.fit(train_data, target)
    score = cross_val_score(SVM_best, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    print(score)
    print("SVM best: ",round(np.mean(score)*100, 2))

    clf = SVC(C=best_C, gamma=best_gamma)
    clf.fit(train_data, target)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })
    #
    submission.to_csv('submission.csv', index=False)
    submission = pd.read_csv('submission.csv')
    submission.head()

# result
def result():
    clf = SVC()
    clf.fit(train_data, target)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
            "PassengerId": test["PassengerId"],
            "Survived": prediction
        })
    #
    # submission.to_csv('submission.csv', index=False)
    # submission = pd.read_csv('submission.csv')
    # submission.head()


RF_Best() # best 4,20  # 0.80382
# AB_Best() # 0.78468
# SVM_Best() # 0.76076