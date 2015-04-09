from __future__ import division, absolute_import, print_function

import numpy
import scipy.sparse
import pyximport
pyximport.install()

from ..Kernels import Kernel
from .FlatProjector import FlatProjector
from . import cy_StackedProjector as cy

class StackedProjector(scipy.sparse.linalg.LinearOperator):
	def __init__(self, kernel, z, shape=None):
		self.z = z

		self.flat_proj = FlatProjector(kernel)

		self.shape = tuple(self.z.size*i for i in self.flat_proj.shape)
		
		self.dtype = self.flat_proj.dtype
		self.dat = self.flat_proj.dat
		self.idx = self.flat_proj.idx
		
	def matvec(self, v):
		v = v.reshape(self.shape[1])
		
		u = numpy.zeros(self.shape[0], self.dtype)

		cy.matvec(v,u, self.dat, self.idx[0], self.idx[1], self.z.size, self.flat_proj.shape[0], self.flat_proj.shape[1])

		return u
	
	def rmatvec(self, v):
		v = v.reshape(self.shape[0])
		
		u = numpy.zeros(self.shape[1], self.dtype)

		cy.matvec(v,u, self.dat, self.idx[1], self.idx[0], self.z.size, self.flat_proj.shape[1], self.flat_proj.shape[0])

		return u