from __future__ import division, absolute_import, print_function

import numpy
import scipy.sparse

class StackProjector(scipy.sparse.linalg.LinearOperator):
	def __init__(self, tensor, z, y, x, d, p):
		self.__dict__.update(dict(tensor=tensor, z=z,y=y,z=z,d=d,t=numpy.require(t)))

		self.shape = (z.size*p.size*d.size, z.size*y.size*x.size)
		
