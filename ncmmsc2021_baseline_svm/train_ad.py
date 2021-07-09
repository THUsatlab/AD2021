#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/5
# @Author  : guyu
import pandas as pd
import os
import numpy as np 
from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,KFold
from numpy import *
from pandas import DataFrame



# 加载训练数据
def load_train_data(train_dataset_path):
    train=pd.read_csv(train_dataset_path)
    features = [str(i) for i in range(1,1583)]
    X_train = train.loc[:,features].values
    Y_train = train.loc[:,'label'].values
    return X_train,Y_train

#预测准确率，查准率，查全率，f1值  
def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance
#进行5次交叉验证
def cross(X,Y):
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
    kf.get_n_splits(X,Y)
    rval0 = []
    rval1 = []
    rval2 = []
    rval3 = []
    for train_index, test_index in kf.split(X, Y):    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        classifier.fit(X_train,y_train)
        #Calculate the score (Accuracy)
        train_score = classifier.score(X_train,y_train)
        test_score = classifier.score(X_test,y_test)
        predict = classifier.predict(X_test)
        p = precision_score(y_test,predict,average='macro')
        r = recall_score(y_test,predict,average='macro')
        f1 = f1_score( y_test,predict,average='macro')
   
        rval0.append(test_score)
        rval1.append(p)
        rval2.append(r)
        rval3.append(f1)
    print('Accuracy rate of five cross-validation:',rval0)
    print('Precision of five cross-validation:',rval1)
    print('Recall of five cross-validation:',rval2)
    print('F1 of five cross-validation:',rval3)
    print('ACC mean:',mean(rval0))
    print('Precision mean:',mean(rval1))
    print('Recall mean:',mean(rval2))
    print('F1 mean:',mean(rval3))
 
        
if __name__ == "__main__":

    # 初始化各个路径
    train_dataset_path='/home/yjj/ME-AD-DATA/ncmmsc2021_baseline_svm/feature/train_1582.csv'
    X_train,Y_train = load_train_data(train_dataset_path)
    #把训练数据归一化
    X_train = StandardScaler().fit_transform(X_train)
    print(X_train.shape)
    #将训练数据划分为训练集和验证集
    X_train_1, X_validation_1, Y_train_1, Y_validation_1 = train_test_split (X_train,Y_train,test_size=0.3,random_state=1)

    #定义分类器
    classifier = svm.SVC(kernel='rbf', probability = True, gamma = 'auto')
    classifier.fit(X_train_1,Y_train_1)
    #估算Accuracy,precision,recall,f1值
    train_score = classifier.score(X_train_1,Y_train_1)
    print('train Accuracy：',train_score)
    predict_validation = classifier.predict(X_validation_1)
    performance = evaluate(Y_validation_1, predict_validation)
    print('validation :',performance)
    #五次交叉验证
    cross(X_train,Y_train)