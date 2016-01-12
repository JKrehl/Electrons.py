from __future__ import division, absolute_import, print_function

import numpy
import scipy.sparse
import pyximport
pyximport.install()

from ..Kernels import Kernel
from .FlatProjector import FlatProjector
from . import StackedProjector_cy as cy

class StackedProjector(scipy.sparse.linalg.LinearOperator):
	def __init__(self, kernel, z, threads=0):
		self.z = z
		self.threads = threads

		if isinstance(kernel, Kernel):

			self.kernel = kernel
			self.shape = list(kernel.fshape)
			self.shape[0] *= self.z.size
			self.shape[1] *= self.z.size

			with kernel.open():
				self.nnz = self.kernel.dat.size
				self.dtype = self.kernel.dat.dtype
		else:
			raise NotImplementedError

		self.matvec = self._matvec
		self.rmatvec = self._rmatvec

	@contextmanager
	def in_memory(self):
		with self.kernel.in_memory('dat', 'row', 'col'):
			yield

	def _matvec(self, v):
		v = v.reshape(self.shape[1])
		
		u = numpy.zeros(self.shape[0], self.dtype)

		with self.in_memory():
			cy.matvec(v,u, self.kernel.dat, self.kernel.row, self.kernel.col, self.z.size, self.kernel.fshape[0], self.kernel.fshape[1], self.threads)

		return u
	
	def _rmatvec(self, v):
		v = v.reshape(self.shape[0])
		
		u = numpy.zeros(self.shape[1], self.dtype)

		with self.in_memory():
			cy.matvec(v,u, self.kernel.dat, self.kernel.col, self.kernel.row, self.z.size, self.kernel.fshape[1], self.kernel.fshape[0], self.threads)

		return u