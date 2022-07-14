from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concept import SimpleConcept,CrispConcept

from  weightfunction import gauss_function

import datetime
import numpy as np



class Structure(object):
    def __init__(self,data):
        """
        data : type is list .
        structure : type is dict.  {(sample_index, Concept): [1,2,...,index],(sample_index, Concept): [1,2,...,index],...}
                    AFS structure saves the index of sample which meets the order relation.
        concepts:  type is set. All simple concept set,
                    e.g.,{SimpleConcept(0, 4.3), SimpleConcept(0, 5.84),...} ,
                    where SimpleConcept is a object, SimpleConcept(feature_index,cut_point)
        weight: type is dict.  weight function value of samples {(sample_index, Concept): value,...}

        weight_sum: type is dict. all samples weight sum at different concept {Concept: sum_value,...}

        """
        self.data = data
        self.structure = {}
        self.concepts = set()
        self.weight = {}
        self.weight_sum = {}

    def generate_structure(self, samples, new_concepts):
        starttime = datetime.datetime.now()
        self.concepts.update(new_concepts) # add new
        # self.data.extend(data)  # add data
        # samples = data  # samples convert to list
        for m in new_concepts:
            weight_val = []
            if m not in self.weight.keys():
                self.weight.setdefault(m, 0.0)  # set default value
            if m not in self.weight_sum.keys():
                self.weight_sum.setdefault(m, 0.0)  # set default value

            for i,val_i in enumerate(samples):
                temp_key = (i, m)
                if temp_key not in self.structure.keys():
                    self.structure.setdefault(temp_key, set())

                #  obatin the weight value by using gauss function
                if isinstance(m, SimpleConcept):
                    # self.weight_sum.update({m: self.weight_sum.get(m) + temp_value})  # update weight_sum
                    booL_list =[]
                    for j, val_j in enumerate(samples):
                        # obtain the index whose value is far away from cut_point_value
                        bool_value = abs(val_i[m.feature_index] - m.cut_point_value) <= abs(val_j[m.feature_index] - m.cut_point_value)
                        booL_list.append(bool_value)
                    self.structure.update({temp_key:booL_list})
                            # self.structure[temp_key].add(j) # update the structure
                elif isinstance(m, CrispConcept):
                    if abs(val_i[m.feature_index] - m.cut_point_value) <= 0.001 : #
                        self.structure[temp_key]= np.ones(len(samples),dtype=bool)
                        self.weight.update({temp_key:1.0})
                        self.weight_sum.update({m:self.weight_sum.get(m)+1})

                temp_value = gauss_function(x=val_i[m.feature_index], mu=m.cut_point_value)

                weight_val.append(temp_value)

            self.weight.update({m: weight_val})  # update weight
            weight_sum = sum(self.weight.get(m))
            self.weight_sum.update({m: weight_sum})
        endtime = datetime.datetime.now()
        print("Endtime", (endtime - starttime).microseconds)

         # If the concepts set is a crisp set ,the weight value can be obtained as follows.
         #        elif isinstance(m, CrispConcept):
         #            if abs(val_i[m.feature_index] - m.cut_point_value) <= 0.001 : #
         #                self.structure[temp_key]={i for i in range(len(samples))}
         #                self.weight.update({temp_key: 1.0})
         #                self.weight_sum.update({m: self.weight_sum.get(m) + 1})
         #        else:
         #            self.weight.update({temp_key: 0.0})
