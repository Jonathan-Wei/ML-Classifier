#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:05:43 2017

@author: tywin
"""



import os
import shutil

import jieba

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import cross_val_score

from sklearn.externals import joblib

stopword_file = open('./stopword.txt')
    
stopword_content_string = stopword_file.read()
    
stopword_list = stopword_content_string.split('\n');
    

'''
第一步，对于中文，需要进行分词处理
'''


# 获取分类
data_list = os.listdir('./data') #语料数据
seg_path = './segment' #分词后的数据
seg_train_path = './segment/train' #分词后的训练数据
seg_test_path = './segment/test'#分词后的测试数据

# 获取各个分类内容，并进行切词保存在segment目录下
def segment_content():
    for data_cate in data_list:
        content = ""
        if not os.path.isdir(data_cate) and not data_cate == '.DS_Store':
            path = './data/%s'%(data_cate)
            files= os.listdir(path)
            for file in files:
                file_path = './data/%s/%s'%(data_cate,file)
                print(file_path)
                if file == ".DS_Store":
                    os.remove(file_path)
                else:
                    f = open(file_path, mode='r', encoding='GB18030')
                    try:
                        content = f.read()
                    except Exception:
                        print('无法读取')
                        continue
                if not os.path.exists('%s/%s'%(seg_train_path,data_cate)):
                    os.mkdir('%s/%s'%(seg_train_path,data_cate))
                output = open('%s/%s/%s'%(seg_train_path,data_cate,file), 'w')
                content_seg = jieba.cut(content) # 使用jieba进行切词
                output.write(" ".join(content_seg))
                output.close()

#从切分的数据中分离出测试数据集
def cut_for_test():
    seg_data_list = os.listdir(seg_train_path)
    for data_dir in seg_data_list:
         
        if os.path.isdir("%s/%s"%(seg_train_path,data_dir)):
            file_dir_path = "%s/%s"%(seg_train_path,data_dir)
            files = os.listdir(file_dir_path)
            test_array = files[-10:]
            for subfile in test_array:
                if subfile == ".DS_Store":
                    os.remove("%s/%s"%(file_dir_path,subfile))
                    continue
                if not os.path.exists("%s/%s"%(seg_test_path,data_dir)):
                    os.mkdir("%s/%s"%(seg_test_path,data_dir))
                shutil.move("%s/%s"%(file_dir_path,subfile),"%s/%s/%s"%(seg_test_path,data_dir,subfile))          
    
    

'''
第二步 获取训练集  返回训练集数据，类别列表
'''
#获取训练数据集
def loadtrainset():
    allfiles = os.listdir(seg_train_path)
    processed_textset  =[]
    allclasstags = []
    for thisdir in allfiles:
        if not thisdir == '.DS_Store':
            
            content_list = os.listdir('%s/%s'%(seg_train_path,thisdir))
            
            for file in content_list:
                if not os.path.isdir('%s/%s/%s'%(seg_train_path,thisdir,file)) and not file == '.DS_Store':
                    path_name = '%s/%s/%s'%(seg_train_path,thisdir,file)
                    print(path_name)
                    f = open(path_name)
                    processed_textset.append(f.read())
                    allclasstags.append(thisdir)
    return processed_textset,allclasstags

#获得测试数据集
def loadtestset():
    allfiles = os.listdir(seg_test_path)
    testset = []
    testclass = []
    
    for cate_dir in allfiles:
        if not cate_dir == '.DS_Store':
            test_content_list = os.listdir('%s/%s'%(seg_test_path,cate_dir))
            for test_file in test_content_list:
                if os.path.isfile('%s/%s/%s'%(seg_test_path,cate_dir,test_file)) and not test_file == '.DS_Store':
                    file_path_name = '%s/%s/%s'%(seg_test_path,cate_dir,test_file)
                    print('获得测试数据文件名')
                    print(file_path_name)
                    f = open(file_path_name)
                    testset.append(f.read())
                    testclass.append(cate_dir)
    return testset,testclass


#第三步，处理词向量，包括去除停用词
def content_vectorizer(data):

    #载入停用词
    # 语料向量化        
    #if not os.path.exists("./vectorizer_content.m"):
    vectorizer_content__ = CountVectorizer(stop_words=stopword_list)
    #vectorizer_content = joblib.load("./vectorizer_content.m")
    x_array = vectorizer_content__.fit_transform(data)
    #print(x_array.toarray())
    # 计算各个分类内容的 tf-idf值
    X_test =  TfidfTransformer().fit_transform(x_array) 
    print('content_vectorizer')
    print(vectorizer_content__)
    joblib.dump(vectorizer_content__, "./vectorizer_content.m") 
    return X_test

#第三步，处理词向量，包括去除停用词
def content_vectorizer_test(data):

    #载入停用词
    # 语料向量化        
    #if not os.path.exists("./vectorizer_content.m"):
    #需要使用和训练器相同的矢量器 否则会报错 ValueError dimension mismatch
    vectorizer_content = joblib.load("./vectorizer_content.m")
    #vectorizer_content = joblib.load("./vectorizer_content.m")
    x_array = vectorizer_content.transform(data)
    #print(x_array.toarray())
    # 计算各个分类内容的 tf-idf值
    X =  TfidfTransformer().fit_transform(x_array) 
    print('content_vectorizer_test')
    print(vectorizer_content)
    #joblib.dump(vectorizer_content__, "./vectorizer_content.m") 
    return X


def file_tools(path):
    content_data = []
    f = open(path)
    content = f.read()
    content_seg = jieba.cut(content)
    string_content = " ".join(content_seg)
    content_data.append(string_content)
    return content_data

# 创建分类器

def train():
    from sklearn.naive_bayes import MultinomialNB
    
    processed_textset,Y = loadtrainset()
    #test_textset,Y_test = loadtestset()
    X_train = content_vectorizer(processed_textset)
    #X_test = content_vectorizer_test(test_textset)
    # 多项式贝叶斯分类器
    clf = MultinomialNB(alpha=0.001).fit(X_train,Y)
    
    # KNN分类器
    #from sklearn.neighbors import KNeighborsClassifier    
    #clf = KNeighborsClassifier().fit(X_train,Y)
    # 随机森林
    #from sklearn.ensemble import RandomForestClassifier    
    #clf = RandomForestClassifier(n_estimators=8)    
    #clf.fit(X_train,Y)  
    
    #predict_result = clf.predict(X_test) #预测结果
    print('交叉验证结果')
    print(cross_val_score(clf,X_train,Y,cv=10,scoring='accuracy'))#交叉验证
    
    #训练完成，保存分类器
    joblib.dump(clf, "./分类器.model") 

#预测
def predict(path):
    check_file_data = file_tools(path)
    check_file_vectorizer = content_vectorizer_test(check_file_data)
    clf = joblib.load("./分类器.model")
    predict_result = clf.predict(check_file_vectorizer)
    print('------>预测结果<------')
    print(predict_result)


#segment_content()
#cut_for_test()
#train()#如果没有训练过模型，需要先通过此方法进行训练
predict('./test2.txt')
