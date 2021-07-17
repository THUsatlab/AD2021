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
from numpy import *
from pandas import DataFrame



# 加载训练数据
def load_train_data(train_dataset_path):
    train=pd.read_csv(train_dataset_path)
    features = [str(i) for i in range(1,1583)]
    X_train = train.loc[:,features].values
    Y_train = train.loc[:,'label'].values
    return X_train,Y_train
# 加载测试数据
def load_test_data(test_dataset_path):
    test=pd.read_csv(test_dataset_path)
    features = [str(i) for i in range(1,1583)]
    X_test = test.loc[:,features].values
    name_test = test.loc[:,'name']
    return X_test,name_test

#预测准确率，查准率，查全率，f1值  
def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance

def change_csv_to_txt(path_csv,path_txt):
    data = pd.read_csv(path_csv, encoding='utf-8')

    with open(path_txt,'a+', encoding='utf-8') as f: 
        f.write(('ID'+' '+'Prediction'+'\n'))
        for line in data.values:
            f.write((str(line[0])+'.wav'+' '+str(line[1])+'\n'))
        
        
if __name__ == "__main__":

    # 初始化各个路径
    train_dataset_path='./feature/train_1582.csv'
    test_dataset_path='./feature/test_1582.csv'
    result_directory_csv='./result/result.csv'
    result_directory_txt='./result/result.txt'
    X_train,Y_train = load_train_data(train_dataset_path)
    X_test,name_test=load_test_data(test_dataset_path)
    
    #把训练数据和测试数据做整体的归一化
    X=np.vstack((X_train,X_test))
    print(X.shape)
    X = StandardScaler().fit_transform(X)
    X_train=X[0:280,:]
    print(X_train.shape)
    X_test=X[280:,]
    print(X_test.shape)
    
    #将训练数据划分为训练集和验证集
    X_train, X_validation, Y_train, Y_validation = train_test_split (X_train,Y_train,test_size=0.3,random_state=1)

    #定义分类器
    classifier = svm.SVC(kernel='rbf', probability = True, gamma = 'auto')
    classifier.fit(X_train,Y_train)
    #估算Accuracy,precision,recall,f1值
    train_score = classifier.score(X_train,Y_train)
    print('train Accuracy：',train_score)
    predict_validation = classifier.predict(X_validation)
    performance = evaluate(Y_validation, predict_validation)
    print('validation :',performance)
    predict_test= classifier.predict(X_test)
    result=DataFrame(columns=['ID','Prediction'])
    result['ID']=name_test
    result['Prediction']=predict_test
    result.to_csv(result_directory_csv ,index=False)
    change_csv_to_txt(result_directory_csv,result_directory_txt)
    
