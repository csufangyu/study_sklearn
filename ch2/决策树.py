# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:32:30 2017

@author: Thinkpad
分类决策树
DecisionTreeClassifier
函数原型为 sklearn.tree.DecisionTreeClassifier(criterion='gini',spiltter='best',
max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,
max_features=None,random_state=None,max_leaf_nodes=None,class_weight=None,presort=False)
每个参数的具体含义如下：
criterion:指定切分质量的评价准则
        'gini'表示切分评价准则是Gini系数
        'entropy'表示切分评价准则是熵
spiltter:指定切分原则
        'best'表示选择最优切分
        'random'表示随机切分
max_depth:可以为整数或者None，指定树的最大深度
         若为None则表示树的深度不限，若max_leaf_nodes非None，则此项忽略
         
min_samples_split:是整数，指定每个内部的节点（非叶节点）包含最小的样本数
min_samples_leaf:是整数，指定每个叶节点包含最小的样本数
min_weight_fraction_leaf：浮点数，叶子节点中样本的最小权重系数
max_features:可以是整数、浮点数、字符串或者None，指定寻找best_split时考虑的特征数量
     如果是整数，则每次划分只考虑max_features个特征
     如果是浮点数，每次划分只考虑max_features*n_features个特征(max_features指定了百分比)
     如果是字符串'auto'或者'sqrt'，则max_features等于sqrt(n_features)
     如果是字符串'log2'，则max_features等于log1(n_features)
     如果是None，则max_features等于n_features
random_state:一个整数或者一个RandomState实例，或者None
      如果是一个整数，则它指定了随机数生成器的种子
      如果为RandomState实例，则指定了随机数生成器
      如果为None，使用默认的随机数生成器
max_leaf_nodes:为整数或者None，指定最大的叶节点数量
      如果为None，此时叶节点数量不限
      如果非None，则max_depth可以忽略
class_weighte:一个字典、字典的列表、字符串'balanced'，或者None,它指定类的权重，形式为
               {class_label:weight}
presort:一个布尔值，指定是否需要提前排序数据，从而加速寻找切分最优切分的过程。为Ture时
       对于大数据集会减慢总体的训练过程，但对于一个小数据集或者指定了最大深度的情况下，
       会加速训练过程
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
'''
这里使用的鸢尾花数据集，共有150个数据，分为三个类，每个类50个，共有4个属性
'''
def load_data():
    iris=datasets.load_iris()
    X_train=iris.data
    y_train=iris.target
    return cross_validation.train_test_split(X_train,y_train,test_size=0.25,random_state=0,
                                        stratify=y_train)
    
def test_DecisionTreeClssifier(*data):
    X_train,X_test,y_train,y_test=data
    clf=DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    print("Training score:%f"%(clf.score(X_train,y_train)))
    print("Testing score:%f"%(clf.score(X_test,y_test)))
    
'''
查看不同的评价准则对于分类效果的影响
'''
def test_DecisionTreeClassifier_criterion(*data):
    X_train,X_test,y_train,y_test=data
    criterions=['gini','entropy']
    for criterion in criterions:
       clf=DecisionTreeClassifier(criterion=criterion)
       clf.fit(X_train,y_train)
       print(criterion)
       print("Training score:%f"%(clf.score(X_train,y_train)))
       print("Testing score:%f"%(clf.score(X_test,y_test))) 
    
'''
查看检验随机划分和最优划分的区别
'''
def test_DecisionTreeClassifier_splitter(*data):
    X_train,X_test,y_train,y_test=data
    splitters=['best','random']
    for splitter in splitters:
        clf=DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train,y_train)
        print(splitter)
        print("Training score:%f"%(clf.score(X_train,y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test))) 
'''
查看深度对于分类的影响
'''
def test_DecisionTreeClassifier_depth(X_train,X_test,y_train,y_test,maxdepth):
    #X_train,X_test,y_train,y_test=data
    depths=np.arange(1,maxdepth)
    training_score=[]
    testing_score=[]
    for depth in depths:
        clf=DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train,y_train)
        training_score.append(clf.score(X_train,y_train))
        testing_score.append(clf.score(X_test,y_test))
   # print(training_score)
    #print(testing_score)
    #绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(depths,training_score,label="training score",marker='o')
    ax.plot(depths,testing_score,label="testing score",marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classifier")
    ax.legend(framealpha=0.5,loc='best')
    plt.show()
        
        
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    #test_DecisionTreeClssifier(X_train,X_test,y_train,y_test)
    #test_DecisionTreeClassifier_criterion( X_train,X_test,y_train,y_test)
    #test_DecisionTreeClassifier_splitter( X_train,X_test,y_train,y_test)
    test_DecisionTreeClassifier_depth( X_train,X_test,y_train,y_test,maxdepth=150)
