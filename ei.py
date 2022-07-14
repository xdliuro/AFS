# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:43:08 2017

@author: Yuangang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import tensorflow as tf
import copy


class EIAlgebra(object):
	def __init__(self):
		return

	""" Return the EI sum of two fuzzy concepts"""

	def ei_sum(self, m, n):
		M = copy.deepcopy(m)
		M.extend(n)
		return self.ei_reduce(M)

	"""Return the EI multiply of two fuzzy concepts"""

	def ei_multiply(self, m, n):
		l = []
		for x in itertools.product(m, n):
			a = set()
			a.update(x[0])
			a.update(x[1])
			l.append(a)
		return self.ei_reduce(l)

	"""Remove the redundant complex concepts"""

	def ei_reduce(self, m):
		r = []
		sorted_m = sorted(m, key=lambda x: len(x))
		for x in sorted_m:
			contain = False
			if len(r) == 0:
				r.append(x)
				continue
			else:
				for y in r:
					if y.issubset(x):
						contain = True
						break
			if contain == False:
				r.append(x)
		return r

	def ei_reduce_contradiction(self, m):
		for concept in m[::-1]:
			conceptlist = list(concept)
			conceptlist.sort()
			for i in range(0, len(conceptlist)-1):
				if conceptlist[i] & 1:
					plus1 = conceptlist[i] + 1
					if plus1 == conceptlist[i+1]:
						if concept in m:
							m.remove(concept)
		return m

	def ei_reduce_contradiction2(self, m):
		lt = []
		for concept in m:
			conceptlist = list(concept)
			conceptlist.sort()
			for i in range(0, len(conceptlist)-1):
				if conceptlist[i] & 1:
					plus1 = conceptlist[i] + 1
					if plus1 == conceptlist[i+1]:
						lt.append(concept)
		for i in lt:
			if i in m:
				m.remove(i)
		return m



	"""Return true if complex concept m < n"""

	def ei_less(self, m, n):
		less = False
		count = 0
		for x in n:
			for y in m:
				if y.issuperset(x):
					count = count + 1
			if count == len(m):
				less = True
				break
		return less

	"""Return true if complex concept m = n"""

	def ei_equal(self, m, n):
		equal = False
		if self.ei_less(m, n) and self.ei_less(n, m):
			equal = True
		return equal

	"""Return negative concept m' """

	def ei_negation(self, m):
		Multiply = []
		for i in m:
			L = []
			for j in list(i):
				# j = c.concept_index
				if (j % 2) == 0: #判断是偶数
					j=j-1
				else: # 是奇数
					j=j+1
				L.append({j})
			Multiply.append(L)
		M = Multiply[0]
		for i in Multiply[1:]:
			M = self.ei_multiply(M, i)
		return M

	"""Return the EI multiply of two fuzzy concepts"""

	def ei_negative_multiply(self, m, n):
		l = []
		for x in itertools.product(m, n):
			a = set()
			a.update(x[0])
			a.update(x[1])
			l.append(a)
		M = self.ei_reduce(l)

		return self.ei_reduce_contradiction(M)

	def Get_value(self, concept, data):
		result = []
		for value in data:
			add = True
			for txt in concept:
				if txt.negation:
					if txt.name not in value[3]:
						add = False
						break
				else:
					if txt.name in value[3]:
						add = False
						break
			if add:
				result.append(value)

	def ei_less_concepts(self, concepts, conceptslist):
		for c in conceptslist:
			if self.ei_less(concepts, [c]):
				return True
		return False


# """Main Function"""
if __name__=="__main__":

	l1 = [{1,  2}]
	# l2 = [{1, 3}, {15, 19}]
	l3 = [{1}]
	# l4 = [{1, 3},{5, 7, 9}]
	# l5 = [{1, 3},{1,3, 5, 7, 9}]
	ei = EIAlgebra()
	# print(ei.ei_negation(l1))
	print(ei.ei_less(l1,l3))
	# print(ei.ei_less(l1,l3))
	# print(ei.ei_equal(l1,l4))
	# print(ei.ei_multiply(l1,l3))
	# print(ei.ei_reduce(l5))
	# print(ei.ei_sum(l1,l2))

