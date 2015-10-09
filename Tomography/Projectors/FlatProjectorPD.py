import numpy
import scipy.sparse

import pyximport
pyximport.install()

from ..Kernels import Kernel
from . import FlatProjectorPD_cy as cy

class FlatProjectorPD(scipy.sparse.linalg.LinearOperator):
	def __init__(self, kernel, shape = None):
		
		if isinstance(kernel, tuple) or isinstance(kernel, list):
			if shape is not None:
				self.shape = shape
			else:
				raise AttributeError("The shape needs to be supplied to the Projector constructor")
			
			dat = kernel[0]
			self.nnz = self.dat.size
			self.dtype = self.dat.dtype
			if len(kernel) == 2:
				assert kernel[1].ndims==2 and kernel[1].shape[0]==2
				idx = tuple(kernel[1])
			elif len(kernel)==3:
				idx = kernel[1:]
			else:
				raise AttributeError
		elif isinstance(kernel, Kernel):
			assert kernel.ndims==2

			if kernel.status == -1:
				kernel.calc()
				
			self.shape = kernel.fshape
			dat = kernel.dat
			idx = kernel.idx
			self.nnz = dat.size
			self.dtype = dat.dtype
		else:
			raise NotImplementedError

		csort = numpy.argsort(idx[1])
		self.c_idxr = idx[0][csort]
		self.c_idxc = numpy.hstack((0, numpy.cumsum(numpy.bincount(idx[1], minlength=self.shape[0])))).astype(idx[1].dtype)
		self.c_dat = dat[csort]

		rsort = numpy.argsort(idx[0])
		self.r_idxc = idx[1][rsort]
		self.r_idxr = numpy.hstack((0, numpy.cumsum(numpy.bincount(idx[0], minlength=self.shape[1])))).astype(idx[0].dtype)
		self.r_dat = dat[rsort]
		

	def matvec(self, v):
		v = v.reshape(self.shape[1])
		
		u = numpy.zeros(self.shape[0], self.dtype)

		cy.matvec(v, u, self.c_dat, self.c_idxc, self.c_idxr)

		return u
	
	def rmatvec(self, v):
		v = v.reshape(self.shape[0])
		
		u = numpy.zeros(self.shape[1], self.dtype)

		cy.matvec(v, u, self.r_dat, self.r_idxr, self.r_idxc)

		return u
