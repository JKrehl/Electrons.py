from __future__ import division, absolute_import, print_function

import numpy

from .Base import PlaneOperator

class SliceStacker(PlaneOperator):
	def __init__(self):
		self.stack = []
		
	def apply(self,wave):
		self.stack.append(wave.copy())
		return wave

	def get(self):
		return numpy.concatenate(tuple(i.reshape(1, *i.shape) for i in self.stack), axis=0)
