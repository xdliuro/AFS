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
from util import Text_to_simpleConcept,transferData
from ClusterBase import sampleDescrib_cluster,build_similarity_matrix,build_idempotent_matrix,each_cluster,cluster_description_function,Re_cluster
import matplotlib.pyplot as plt
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

"""=========建立Structure结构==========================="""
str = Structure(train_data,copy_onehot_target)

"""========在Structure中添加简单概念====================="""
# conceptM 需要添加的概念
str.generate_structure(conceptM)
# print(str.structure)
"""=========做EI代数运算================================="""
ei = EIAlgebra()

"""=========求隶属度函数================================="""
mf = MembershipFunction()
# mf.show_membership_of_concepts_alldata(str, conceptSet1)

sample_description = sampleDescrib_cluster(ei, mf, str, 1, epsilon = 0.3)
print(sample_description)

print("Clustering SampleDescription")
sample_description_cluster= []
for i in range(0,len(str.data)):
	sample_description_cluster.append(sampleDescrib_cluster(ei, mf, str, i, epsilon=0.3))
print(sample_description_cluster)

"=============build_similarity_matrix======================"
similarity_matrix = build_similarity_matrix(ei, mf, str,sample_description_cluster)
"=============build idempotent matrix===================================="
idempotent_matrix = build_idempotent_matrix(similarity_matrix)
"============each_cluster============================="
cluster  = each_cluster(idempotent_matrix,0.8)
"""==========each_cluster_description============================="""
cluster_desciprion = cluster_description_function(ei,cluster,sample_description_cluster)

print("cluster_description")
print(cluster_desciprion)

"""=========Recluster description===================== """
re_cluster = Re_cluster(mf,str,cluster_desciprion)
accuracy = sum(integer_encoded==re_cluster)/len(str.data)
print(accuracy)

"===========相似性矩阵==============="
plt.figure()
plt.matshow(similarity_matrix)
plt.title("similarity_matrix")
plt.colorbar()








