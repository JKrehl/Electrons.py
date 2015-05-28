from __future__ import division, absolute_import, print_function

import numpy
import scipy.sparse
import pyximport
pyximport.install()

from ..Kernels import Kernel
from . import StackedExtendedProjector_cy as cy

class StackedExtendedProjector(scipy.sparse.linalg.LinearOperator):
	def __init__(self, kernel, z, shape=None, threads=0):
		self.z = z

		self.kernel = kernel

		if shape is not None:
			self.shape = shape
		else:
			self.shape = tuple(i//self.kernel.idz.size*self.z.size for i in self.kernel.fshape)

		self.dtype = self.kernel.dat.dtype
		self.itype = self.kernel.idx[0].dtype

		self.dat = self.kernel.dat
		self.idx = self.kernel.idx
		self.idz = self.kernel.idz
		self.bounds = self.kernel.bounds
		
		self.threads = threads
		
	def matvec(self, v):
		v = v.reshape(self.shape[1])
		
		u = numpy.zeros(self.shape[0], self.dtype)

		cy.matvec(v,u, self.dat, self.idz, self.bounds, self.idx[1], self.idx[0], self.z.size, self.threads)

		return u
