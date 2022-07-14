# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:52:30 2017

@author: Wenjuan Jia

You can add different weight function

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

def gauss_function(x,mu=0.5,sigma=1):
	gauss_value = math.exp((-0.5 * (x - mu) ** 2)/sigma**2)
	return gauss_value

