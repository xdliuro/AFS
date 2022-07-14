# -*- coding: utf-8 -*-
"""
Created on Sat 2019.10.23

@author: Wenjuan Jia
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concept import SimpleConcept
from structure_bool import Structure
from membership_function_bool import MembershipFunction
from ei import EIAlgebra
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from util import Text_to_simpleConcept,transferData
from ClassifierBase import sampleDescrib_classify,each_class_description, Re_class

import numpy as np

"""======Load data==================================="""
# 参考示例使用的是arff文件
data_set, meta= arff.loadarff('iris.arff')

train_data, copy_onehot_target,integer_encoded = transferData(data_set, meta)

"""=====对文本框输入参数进行处理，生成简单概念==============="""
# 参数设定指定简单概念，奇数概念表示肯定概念，偶数表示否定概念
# Iris有4个feature，（1,3,5）,（7,9,11）,（13,15,17）,（19,21,23）表示每个属性上的简单概念大中小
# x的设置方式 （13 2 1）:第三个属性上的简单肯定概念小；
# 13：第三个属性上的简单概念，2：第三个属性（python下标从0开始）1：第三个属性上的最小值

x = "13 2 1\n15 2 3.75\n17 2 6.9\n19 3 0.1\n21 3 1.19\n23 3 2.5\n"
simpleConceptSet = Text_to_simpleConcept(x)
print(simpleConceptSet)

"""======生成concept集合============================="""
conceptM = {}
for i in simpleConceptSet:
	conceptM.update({int(i[0]):SimpleConcept(int(i[0]),int(i[1]),i[2])})
print(conceptM)

"""=====建立Structure结构==========================="""
str = Structure(train_data,copy_onehot_target)

"""====在Structure中添加简单概念====================="""
# conceptM 需要添加的概念
str.generate_structure(conceptM)
# print(str.structure)
"""=====做EI代数运算================================="""
ei = EIAlgebra()

"""=====求隶属度函数================================="""
mf = MembershipFunction()
# mf.show_membership_of_concepts_alldata(str, conceptSet1)

"""=====计算所有样本的描述（分类）==================="""
sample_description_classify=[]
for i in range(0,len(str.data)):
	sample_description_classify.append(sampleDescrib_classify(ei, mf, str, i, str.target, epsilon=0.3,threshold1=0.5,threshold2=0.5))
copy_description = np.transpose(np.tile(sample_description_classify, (np.array(str.target).shape[1],1)))
description_set = np.multiply(copy_description,np.array(str.target).astype(int))

"""======计算每类的类描述============================"""
class_description = []
for i in range(0,copy_onehot_target.shape[1]):
	class_description.append(each_class_description(description_set[:,i]))

mf.show_membership_of_concepts_alldata(str,class_description[0],plot=True)
mf.show_membership_of_concepts_alldata(str,class_description[1],plot=True)
mf.show_membership_of_concepts_alldata(str,class_description[2],plot=True)

print("Class_description")
print(class_description)


re_class = Re_class(mf,str,class_description)
accuracy = sum(integer_encoded==re_class)/len(str.data)
print(accuracy)



