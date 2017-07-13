# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:27:15 2017

@author: Thinkpad
主要是介绍贝叶斯分类器
1.高斯贝叶斯分类器
class sklearn.naive_bayes.GaussianNB
高斯贝叶斯分类器没有参数

2.多项式贝叶斯分类器
class sklearn.naive_bayes.MultinomialNB(alpha=1.0,fit_prior=True,class_prior=None)
参数含义如下：
alpha：一个浮点数，指定alpha的值
fit_prior:布尔值，如果为Ture，则不用去学习P(y=ck),以均匀分布替代，否则则去学习P（y=ck）
class_prior:一个数组。它指定了每个分类的先验概率P(y=c1),P(y=c2).....,若指定了该参数
            则每个分类的先验概率无需学习
伯努利贝叶斯分类器
class sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=0.0,fit_prior=Ture,
class_prior=None)
参数含义如下：
alpha：一个浮点数，指定alpha的值
binarize:一个浮点数或者None
       如果为浮点数则以该数值为界，特征值大于它的取1，小于的为0
       如果为None，假定原始数据已经二值化
fit_prior:布尔值，如果为Ture，则不用去学习P(y=ck),以均匀分布替代，否则则去学习P（y=ck）
class_prior:一个数组。它指定了每个分类的先验概率P(y=c1),P(y=c2).....,若指定了该参数
            则每个分类的先验概率无需学习       

"""
from sklearn import datasets,cross_validation,naive_bayes
import numpy as np
import matplotlib.pyplot as plt

def show_digits():
    digits=datasets.load_digits()
    fig=plt.figure()
    print('vector from image 0:',digits.data[0])
    for i in range(25):
        ax=fig.add_subplot(5,5,i+1)
        ax.imshow(digits.images[i],cmap=plt.cm.gray_r,interpolation='nearest')
    plt.show()
    
def load_data():
    digits=datasets.load_digits()
    return cross_validation.train_test_split(digits.data,digits.target,test_size=0.25,
                                             random_state=0)

#用高斯分类器来查看效果
def test_GaussianNB(*data):
    X_train,X_test,y_train,y_test=data
    cls=naive_bayes.GaussianNB()
    cls.fit(X_train,y_train)
    print("training score:%.2f"%(cls.score(X_train,y_train)))
    print("testing score:%.2f"%(cls.score(X_test,y_test)))
    
#测试多项式贝叶斯分类器
def test_MultinomialNB(*data):
    X_train,X_test,y_train,y_test=data
    cls=naive_bayes.MultinomialNB()
    cls.fit(X_train,y_train)
    print("training score:%.2f"%(cls.score(X_train,y_train)))
    print("testing score:%.2f"%(cls.score(X_test,y_test)))
#检验不同alpha值对于分类结果的影响
def test_MultinomialNB_alpha(*data):
    X_train,X_test,y_train,y_test=data
    alphas=np.logspace(-2,5,num=200)
    training_score=[]
    testing_score=[]
    for alpha in alphas:
        cls=naive_bayes.MultinomialNB(alpha=alpha)
        cls.fit(X_train,y_train)
        training_score.append(cls.score(X_train,y_train))
        testing_score.append(cls.score(X_test,y_test))
        
    #绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,training_score,label="training score")
    ax.plot(alphas,testing_score,label="testing score")
    ax.set_xlabel('alpha')
    ax.set_ylabel('score')
    ax.set_title("MultinomoalNB")
    ax.set_xscale("log")
    plt.show()
 #查看伯努利分类器效果   
def test_BernoulliNB(*data):
    X_train,X_test,y_train,y_test=data
    cls=naive_bayes.BernoulliNB()
    cls.fit(X_train,y_train)
    print("training score:%.2f"%(cls.score(X_train,y_train)))
    print("testing score:%.2f"%(cls.score(X_test,y_test)))
    
 
## 查看不同alpha值的影响
def test_BernoulliNB_alpha(*data):
    X_train,X_test,y_train,y_test=data
    alphas=np.logspace(-2,5,num=200)
    training_score=[]
    testing_score=[]
    for alpha in alphas:
        cls=naive_bayes.BernoulliNB(alpha=alpha)
        cls.fit(X_train,y_train)
        training_score.append(cls.score(X_train,y_train))
        testing_score.append(cls.score(X_test,y_test))
        
    #绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,training_score,label="training score")
    ax.plot(alphas,testing_score,label="testing score")
    ax.set_xlabel('alpha')
    ax.set_ylabel('score')
    ax.set_title("BerbuonlliNB")
    ax.set_xscale("log")
    plt.show()
 ##查看不同阙值的影响
def test_BernoulliNB_binarize(*data):
    X_train,X_test,y_train,y_test=data
    min_x=min(np.min(X_train.ravel()),np.min(X_test.ravel()))-0.1
    max_x=max(np.max(X_train.ravel()),np.max(X_test.ravel()))-0.1
    binarizes=np.linspace(min_x,max_x,endpoint=True,num=100)
    training_score=[]
    testing_score=[]
    for binarize in binarizes:
        cls=naive_bayes.BernoulliNB(binarize=binarize)
        cls.fit(X_train,y_train)
        training_score.append(cls.score(X_train,y_train))
        testing_score.append(cls.score(X_test,y_test))
    ##绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(binarizes,training_score,label="training score")
    ax.plot(binarizes,testing_score,label="testing score")
    ax.set_xlabel('binarize')
    ax.set_ylabel('score')
    ax.set_title("BerbuonlliNB")
    plt.show()

if __name__=="__main__":
    #show_digits()
     X_train,X_test,y_train,y_test=load_data()
     #test_GaussianNB(X_train,X_test,y_train,y_test)
     #test_MultinomialNB(X_train,X_test,y_train,y_test)
     #test_MultinomialNB_alpha(X_train,X_test,y_train,y_test)
     #test_BernoulliNB_alpha(X_train,X_test,y_train,y_test)
     test_BernoulliNB_binarize(X_train,X_test,y_train,y_test)
