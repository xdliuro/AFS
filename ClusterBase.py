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


"""===============================build_similarity_matrix====================== """
def build_similarity_matrix(ei, mf, str,sample_description_cluster):
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


"""=====================each_cluster_description============================="""
def cluster_description_function(ei,cluster,sample_description_cluster):
	cluster_description_list = []
	for i in cluster:
		Temp =[]
		for j in i:
			Temp= ei.ei_sum(Temp,sample_description_cluster[j])
		cluster_description_list.append(Temp)

	return cluster_description_list


"""==================Recluster description===================== """

def Re_cluster(mf,str,cluster_desciprion):
	data_degree = []
	for i in cluster_desciprion:
		data_degree.append(mf.show_membership_of_concepts_alldata(str, i))

	re_cluster = np.argmax(data_degree,axis=0)
	return re_cluster