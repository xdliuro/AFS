# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:52:30 2017

@author: Yuangang Wang
@revise: Hongyue Guo
@ author：Wenjuan Jia
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import datetime

class MembershipFunction(object):

    """Initialization"""
    def __init__(self):

        pass


    """Calculate the membership degree of simple concept"""

    def get_membership_degree_of_SimpleConcept(self, concept, structure):
        degree=[]
        Weight_mat = np.array(structure.weight.get(concept))
        for sample_index in range(len(structure.data)):
            Structure_mat = np.array(structure.structure.get((sample_index, concept)))
            degree.append(np.sum(np.multiply(Weight_mat,Structure_mat))/(structure.weight_sum.get(concept) + 0.00000000001))
        return degree


    """Calculate the membership degree of complex concept with 'and' operation"""
    def get_membership_degree_wedge(self, concept, sample_index, structure):
        Mat = np.ones(len(structure.data),dtype=bool)
        degree = 1
        for m in concept:
            # and_set   the index of conjunction concepts
            Structure_mat = np.array(structure.structure.get((sample_index, m)))
            Mat= np.logical_and(Mat,Structure_mat)
        for m in concept:
            Weight_mat = np.array(structure.weight.get(m))
            SumDegree = np.sum(np.multiply(Mat,Weight_mat))/(structure.weight_sum.get(m) + 0.00000000001)
            degree = degree*SumDegree

        return degree


    """Calculate the membership degree of complex concept with 'and' and 'or' operation"""

    def get_membership_degree_wedge_vee(self, concept, sample_index, structure):
        temp = []
        for i , val in enumerate(concept):
            temp.append(self.get_membership_degree_wedge(val, sample_index, structure))

        Or_membership_degree = max(temp)

        return Or_membership_degree

    """Show distribution of membership on a concept"""

    """Show membership of all data on complex concepts"""
    def show_membership_of_concepts_alldata(self, str, concepts,plot=False):
        # starttime = datetime.datetime.now()
        degrees = []
        for i in range(len(str.data)):
            degrees.append(self.get_membership_degree_wedge_vee(concepts, i, str))
        # endtime = datetime.datetime.now()
        # print("generate_membership_time", (endtime - starttime).microseconds)
        if plot:
            plt.figure()
            plt.plot(degrees)
            plt.show()
            # plt.savefig("AFS.png")
        return degrees