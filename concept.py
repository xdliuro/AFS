# -*- coding: utf-8 -*-
"""
Created on Sat Dec 2 20:25:17 2017

@author: Yuangang Wang
@revise: Wenjuan Jia
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class SimpleConcept(object):
    
    """Initialization
     initial simple concept
     feature_index : the feature index of data
     cut_point_value : the cut point value of each feature

    """
    def  __init__(self, concept_index, feature_index, cut_point_value, subpreference= None):
        self.concept_index = concept_index
        self.cut_point_value = cut_point_value
        self.feature_index = feature_index
        self.subpreference = subpreference

        
    """Return a string for describing the object"""
    def __repr__(self):
        return "SimpleConcept(%s ,%s, %s)" % (self.concept_index, self.feature_index, self.cut_point_value)
    
        
    """Determine whether two simple concepts are equal"""
    def __eq__(self, other):
        if isinstance(other, SimpleConcept):
            return ((self.feature_index == other.feature_index) and (self.cut_point_value == other.cut_point_value))
        else:
            return False

    """Obtain the hash value of the object"""
    def __hash__(self):
        return hash(self.__repr__())


class CrispConcept(object):
    """Initialization
    initial crisp concept: the type of feature is a crisp concept
    feature_index : the feature index of data
    cut_point_value : the cut point value of each feature
    """

    def __init__(self, concept_index,feature_index, cut_point_value):
        self.concept_index = concept_index
        self.feature_index = feature_index
        self.cut_point_value = cut_point_value

    """Return a string for describing the object"""

    def __repr__(self):
        return "CrispConcept(%s, %s)" % (self.feature_index, self.cut_point_value)

    """Determine whether two simple concepts are equal"""

    def __eq__(self, other):
        if isinstance(other, SimpleConcept):
            return ((self.feature_index == other.feature_index) and (self.cut_point_value == other.cut_point_value))
        else:
            return False

    """Obtain the hash value of the object"""

    def __hash__(self):
        return hash(self.__repr__())
