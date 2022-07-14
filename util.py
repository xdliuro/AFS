from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import copy
#  load the data : the type of data is arff
def load_data(file_name):
	data, meta = arff.loadarff(file_name)
	return  data, meta


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


def transferData(data_set, meta):
	# Converting to a normal array
	In_train_data =np.array(data_set[meta.names()[:-1]])
	train_data = copy.deepcopy(In_train_data.tolist())
	target = np.array(data_set[meta.names()[-1]])

	# 对target进行onehot编码
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(target)
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded_label = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_target = onehot_encoder.fit_transform(integer_encoded_label)
	copy_onehot_target = copy.deepcopy(onehot_target)

	return train_data, copy_onehot_target,integer_encoded