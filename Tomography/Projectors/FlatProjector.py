from __future__ import division, absolute_import, print_function

import numpy
import scipy.sparse
import pyximport
pyximport.install()

from ..Kernels import Kernel
from . import cy_FlatProjector as cy

class FlatProjector(scipy.sparse.linalg.LinearOperator):
	def __init__(self, kernel, shape = None):
		
		if isinstance(kernel, tuple) or isinstance(kernel, list):
			if shape is not None:
				self.shape = shape
			else:
				raise AttributeError("The shape needs to be supplied to the Projector constructor")
			
			self.dat = kernel[0]
			self.nnz = self.dat.size
			self.dtype = self.dat.dtype
			if len(kernel) == 2:
				assert kernel[1].ndims==2 and kernel[1].shape[0]==2
				self.idx = tuple(kernel[1])
			elif len(kernel)==3:
				self.idx = kernel[1:]
			else:
				raise AttributeError
		elif isinstance(kernel, Kernel):
			assert kernel.ndims==2

			if kernel.status == -1:
				kernel.calc()
				
			self.shape = kernel.fshape
			self.dat = kernel.dat
			self.idx = kernel.idx
			self.nnz = self.dat.size
			self.dtype = self.dat.dtype
		else:
			raise NotImplemented

	def matvec(self, v):
		v = v.reshape(self.shape[1])
		
		u = numpy.zeros(self.shape[0], self.dtype)

		cy.matvec(self.dat, self.idx[0], self.idx[1], v, u, self.dat.size)

		return u
	
	def rmatvec(self, v):
		v = v.reshape(self.shape[0])
		
		u = numpy.zeros(self.shape[1], self.dtype)

		cy.rmatvec(self.dat, self.idx[0], self.idx[1], v, u, self.dat.size)

		return u
