# -*- coding: utf-8 -*-
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import CollectiveClassifier

f = open('单细胞测序数据.csv')
df = pd.read_csv(f)
feature = df.values.tolist()
f.close()
f = open('单细胞测序数据Labels.csv')
df = pd.read_csv(f, header=None)
label = df.values.tolist()
ytem = []
for i in label:
    ytem.append(i[0])
label = ytem
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.3, stratify=label)
# 标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
collectivemodel = CollectiveClassifier.CollectiveClassifier()
collectivemodel.fit(X_train,Y_train)
collectivemodel.score(X_test,Y_test)

