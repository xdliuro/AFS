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
from util import Text_to_simpleConcept


import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt


"""=======================================sampleDescrib_classify============================================================="""

def sampleDescrib_classify(ei, mf, str, sample_index, onehot_target, epsilon = 0.3, threshold1=0.5,threshold2=0.5):
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


"""=================================计算每个类的类描述 ============================================================="""
def each_class_description(description_list):
	description =[]
	for i in description_list:
		for j in i:
			if j in description:
				continue
			else:
				description.append(j)
	return description


def Re_class(mf,str,cluster_desciprion):
	data_degree = []
	for i in cluster_desciprion:
		data_degree.append(mf.show_membership_of_concepts_alldata(str, i))

	re_cluster = np.argmax(data_degree,axis=0)
	return re_cluster

