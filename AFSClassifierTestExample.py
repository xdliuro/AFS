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

import numpy as np
import datetime
import copy
import pickle
import itertools
import matplotlib.pyplot as plt

"""===================Load data=========================="""
# doc = open("AFSmain.txt","w")

starttime = datetime.datetime.now()
data_set, meta= arff.loadarff('iris.arff')

# Converting to a normal array
In_train_data =np.array(data_set[meta.names()[:-1]])
# In_train_data = In_train_data.view(np.float64).reshape(In_train_data.shape + (-1,))
train_data1 = copy.deepcopy(In_train_data.tolist())
train_data2 = copy.deepcopy(In_train_data.tolist())
target = np.array(data_set[meta.names()[-1]])

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(target)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded_label = integer_encoded.reshape(len(integer_encoded), 1)
onehot_target = onehot_encoder.fit_transform(integer_encoded_label)
copy_onehot_target1 = copy.deepcopy(onehot_target)
copy_onehot_target2 = copy.deepcopy(onehot_target)
# print(copy_onehot_target)

# parameter = [[4.3, 7.9], [2.0,  4.4], [1.0,  6.9], [0.1, 2.5]]
# m1 = SimpleConcept(1, 0, 7.9)
# m2 = SimpleConcept(3, 0, 4.3)
# m3 = SimpleConcept(5, 2, 1.5)
# m4 = SimpleConcept(7, 2, 2)
# m5 = SimpleConcept(9, 2, 6.9)
# m6 = SimpleConcept(11,3,2.5)
# conceptM = [m1, m2, m3, m4, m5, m6]

"""======================对文本框输入参数进行处理，生成简单概念==============="""
# parameter = [[4.3, 5.8, 7.9], [2.0, 3.05, 4.4], [1.0, 3.75, 6.9], [0.1, 1.19, 2.5]]
# x = "1 0 4.3\n3 0 5.8\n5 0 7.9\n7 1 2.0\n9 1 3.05\n11 1 4.4\n13 2 1\n15 2 3.75\n17 2 6.9\n19 3 0.1\n21 3 1.19\n23 3 2.5\n"
# x = "12 2 1.0\n13 2 1.0\n19 3 0.1\n"
x = "13 2 1\n15 2 3.75\n17 2 6.9\n19 3 0.1\n21 3 1.19\n23 3 2.5\n"
def Text_to_simpleConcept(x):
	L = x.split("\n")
	Temp = []
	for i in L:
		if i:
			temp1 = i.split(" ")
			temp2 = []
			for j in temp1:
				temp2.append(float(j))
			Temp.append(temp2)
	return Temp

simpleConceptSet = Text_to_simpleConcept(x)
print(simpleConceptSet)

"""=========================生成concept集合====================================="""
# conceptM = [SimpleConcept(int(i[0]),int(i[1]),i[2]) for i in simpleConceptSet]
# print(conceptM)
conceptM = {}
for i in simpleConceptSet:
	conceptM.update({int(i[0]):SimpleConcept(int(i[0]),int(i[1]),i[2])})
print(conceptM)

"""==========================建立Structure结构========================================"""
str = Structure(train_data1,copy_onehot_target1)
# """generate AFS structure"""

"""==============================增加概念============================================="""
# conceptM 需要添加的概念
str.generate_structure(conceptM)
# print(str.structure)


"""==============================删除概念==========================================="""
# index 需要删除的概念标号
# str.delt_concept(Index)
# print(str.concepts)

"""==============================增加数据============================================="""
# train_data2 需要添加的数据，opy_onehot_target2 新添数据的标号
# str.update_structure(train_data2,copy_onehot_target2)
# print(str.structure)

"""==============================删除数据=============================================="""
#  Index 要删除的数据
# str.delt_data(Index)


"""============================做EI代数运算================================="""
ei = EIAlgebra()


"""=============================求隶属度函数============================================="""
mf = MembershipFunction()
# mf.show_membership_of_concepts_alldata(str, conceptSet1)


"""============================sampleDescrib_cluster============================================================="""

#生成样本的描述
def sampleDescrib_cluster(ei, mf, str, sample_index, epsilon = 0.3):
	simple_concept_objects = set(str.concepts.values())
	sum_description = []
	product_description = set()
	for i in simple_concept_objects:
		sum_description.append(set([i]))
		product_description.add(i)
	sum_degree = mf.get_membership_degree_wedge_vee(sum_description, sample_index, str)
	# print (sum_degree)
	exclude_description = []
	# exclude_description.append(product_description)
	sample_description = []
	for i in range(1, len(simple_concept_objects)+1):
		for j in itertools.combinations(simple_concept_objects, i):
			description = set()
			for k in j:
				description.add(k)
			# print(description)
			# if len(exclude_description) != 0 and ei.ei_less([set(j)], exclude_description, ):
			if len(exclude_description)!=0 and ei.ei_less([description],exclude_description,):
				continue
			else:
				degree = mf.get_membership_degree_wedge(description, sample_index, str)
				# print(degree)
				if degree < sum_degree - epsilon:
					exclude_description.append(description)
				else:
					sample_description.append(description)
	return sample_description

sample_description = sampleDescrib_cluster(ei, mf, str, 1, epsilon = 0.3)
print(sample_description)



print("Clustering SampleDescription")
sample_description_cluster= []
for i in range(0,len(str.data)):
	sample_description_cluster.append(sampleDescrib_cluster(ei, mf, str, i, epsilon=0.3))
print(sample_description_cluster)

# mf.show_membership_of_concepts_alldata(str,sample_description,plot=True)
# mf.show_membership_of_concepts_alldata(str,[sample_description[0]],plot=True)
# mf.show_membership_of_concepts_alldata(str,[sample_description[1]],plot=True)


# def build_similarity_matrix(ei, mf, str):
# 	similarity = np.zeros((len(str.data), len(str.data)))
# 	for i in range(0, len(str.data)):
# 		for j in range(i, len(str.data)):
# 			description_i = sampleDescrib_cluster(ei, mf, str, i)
# 			description_j = sampleDescrib_cluster(ei, mf, str, j)
# 			description_i_and_j = ei.ei_multiply(description_i, description_j)
# 			degree_i = mf.get_membership_degree_wedge_vee(description_i_and_j, i, str)
# 			degree_j = mf.get_membership_degree_wedge_vee(description_i_and_j, j, str)
# 			sim = 0
# 			if degree_i <= degree_j:
# 				sim = degree_i
# 			else:
# 				sim = degree_j
# 			similarity[i, j] = sim
# 			similarity[j, i] = sim
# 			print (i, j)
# 	return similarity

"""===============================build_similarity_matrix====================== """
def build_similarity_matrix(ei, mf, str):
	similarity = np.zeros((len(str.data), len(str.data)))
	for i in range(0, len(str.data)):
		for j in range(i, len(str.data)):
			description_i = sample_description_cluster[i]
			description_j = sample_description_cluster[j]
			description_i_and_j = ei.ei_multiply(description_i, description_j)
			degree_i = mf.get_membership_degree_wedge_vee(description_i_and_j, i, str)
			degree_j = mf.get_membership_degree_wedge_vee(description_i_and_j, j, str)
			sim = min(degree_i,degree_j)
			similarity[i, j] = sim
			similarity[j, i] = sim
			# print(i, j)
	return similarity

similarity_matrix = build_similarity_matrix(ei, mf, str)
plt.figure()
plt.matshow(similarity_matrix)
plt.title("similarity_matrix")
plt.colorbar()

"""==============================build idempotent matrix====================================="""
def build_idempotent_matrix(similarity_matrix):
	similarity_matrix_test = similarity_matrix
	for k1 in range(0,len(similarity_matrix_test)):
		idempotent_matrix_test = np.zeros((len(similarity_matrix_test), len(similarity_matrix_test)))
		for i in range(0,len(similarity_matrix_test)):
			for j in range(0,len(similarity_matrix_test)):
				for k2 in range(0,len(similarity_matrix_test)):
					idempotent_matrix_test[i,j]= max(idempotent_matrix_test[i,j],min(similarity_matrix_test[i,k2],similarity_matrix_test[k2,j]))

		test = sum(sum(idempotent_matrix_test==similarity_matrix_test))
		if test==len(similarity_matrix_test)*len(similarity_matrix_test):
			break
		similarity_matrix_test=idempotent_matrix_test

	idempotent_matrix = similarity_matrix_test
	return idempotent_matrix


idempotent_matrix = build_idempotent_matrix(similarity_matrix)

# print(idempotent_matrix)
plt.figure()
plt.matshow(idempotent_matrix)
plt.title("idempotent_matrix")
plt.colorbar()


"""=====================each_cluster============================="""


def each_cluster(idempotent_matrix,alpha):
	index = np.argwhere(idempotent_matrix>alpha).tolist()
	cluster = []
	for i in range(0,len(index)):
		if cluster:
			flag = True
			for cluster_index in cluster:
				if set(cluster_index).intersection(index[i]):
					cluster_index.update(set(index[i]))
					flag = False
				else:
					continue
			if flag:
				cluster.append(set(index[i]))
		else:
			cluster.append(set(index[i]))
	return cluster


cluster  = each_cluster(idempotent_matrix,0.8)

"""=====================each_cluster_description============================="""
def cluster_description_function(cluster):
	cluster_description_list = []
	for i in cluster:
		Temp =[]
		for j in i:
			Temp= ei.ei_sum(Temp,sample_description_cluster[j])
		cluster_description_list.append(Temp)

	return cluster_description_list

cluster_desciprion = cluster_description_function(cluster)

print("cluster_description")
print(cluster_desciprion)


"""==================Recluster description===================== """

def Re_cluster(mf,str,cluster_desciprion):
	data_degree = []
	for i in cluster_desciprion:
		data_degree.append(mf.show_membership_of_concepts_alldata(str, i))

	re_cluster = np.argmax(data_degree,axis=0)
	return re_cluster

re_cluster = Re_cluster(mf,str,cluster_desciprion)

accuracy = sum(integer_encoded==re_cluster)/len(str.data)
print(accuracy)

"""=======================================sampleDescrib_classify============================================================="""

def sampleDescrib_classify(ei, mf, str, sample_index, onehot_target, epsilon = 0.3,threshold1=0.5,threshold2=0.5):
	simple_concept_objects = str.concepts.values()
	sum_description = []
	product_description = set()
	for i in simple_concept_objects:
		sum_description.append(set([i]))
		product_description.add(i)
	sum_degree = mf.get_membership_degree_wedge_vee(sum_description, sample_index, str)
	# print (sum_degree)
	exclude_description = []
	# exclude_description.append(product_description)
	sample_description = []
	for i in range(1, len(simple_concept_objects)+1):
		for j in itertools.combinations(simple_concept_objects, i):
			description = set()
			for k in j:
				description.add(k)
			# print(description)
			# if len(exclude_description) != 0 and ei.ei_less([set(j)], exclude_description, ):
			if len(exclude_description)!=0 and ei.ei_less([description],exclude_description,):
				continue
			else:
				degree = mf.get_membership_degree_wedge(description, sample_index, str)
				# print(degree)
				if degree < sum_degree - epsilon:
					exclude_description.append(description)
				else:
					data_degree = mf.show_membership_of_concepts_alldata(str,[description])
					copydata= np.tile(data_degree,  (np.array(str.target).shape[1],1))
					copy_data_degree = np.transpose(copydata)
					each_sample_MF = np.multiply(copy_data_degree,np.array(onehot_target).astype(int))
					Mean = np.sum(each_sample_MF,axis=0)/np.sum(np.array(onehot_target),axis=0)
					# print(Mean)
					sample_label = onehot_target[sample_index]
					iner_membership = np.sum(np.multiply(Mean,sample_label))
					ex_membership = np.sum(np.multiply(Mean,np.logical_not(sample_label)))
					# iner_membership = np.sum(np.mean(each_sample_MF[0:50],axis=0))
					# ex_membership = np.sum(np.mean(each_sample_MF[50:150],axis=0))
					if iner_membership>threshold1 and ex_membership<threshold2:
						sample_description.append(description)
					else:
						exclude_description.append(description)

	return sample_description

"""==========================================================================================="""

# print("Classifiy SampleDescription")
sample_description_classify = sampleDescrib_classify(ei, mf, str, 1, str.target,epsilon = 0.3)
# print(sample_description_classify)
# mf.show_membership_of_concepts_alldata(str,sample_description_classify,plot=True)


"""=================================Class description ============================================================="""
def each_class_description(description_list):
	description =[]
	for i in description_list:
		for j in i:
			if j in description:
				continue
			else:
				description.append(j)
	return description

"""================================计算所有样本的描述================================="""
sample_description_classify=[]
for i in range(0,len(str.data)):
	sample_description_classify.append(sampleDescrib_classify(ei, mf, str, i, str.target, epsilon=0.3))
# print(sample_description_classify)
copy_description = np.transpose(np.tile(sample_description_classify, (np.array(str.target).shape[1],1)))
description_set = np.multiply(copy_description,np.array(str.target).astype(int))
"""================================计算每类样本的描述================================="""
class_description = []
for i in range(0,onehot_target.shape[1]):
	class_description.append(each_class_description(description_set[:,i]))

print("Class_description")
print(class_description)

mf.show_membership_of_concepts_alldata(str,class_description[0],plot=True)
mf.show_membership_of_concepts_alldata(str,class_description[1],plot=True)
mf.show_membership_of_concepts_alldata(str,class_description[2],plot=True)


def Re_class(mf,str,cluster_desciprion):
	data_degree = []
	for i in cluster_desciprion:
		data_degree.append(mf.show_membership_of_concepts_alldata(str, i))

	re_cluster = np.argmax(data_degree,axis=0)
	return re_cluster

re_cluster = Re_class(mf,str,class_description)

accuracy = sum(integer_encoded==re_cluster)/len(str.data)
print(accuracy)



# Pickle dictionary using protocol 0
# output = open('AFSmodel.pkl', 'wb')
# pickle.dump(class_description,output)
#
# output.close()

# pkl_file = open('AFSmodel.pkl', 'rb')
#
# class_description1 = pickle.load(pkl_file)
# print(class_description1)


