import numpy
import scipy.sparse

import pyximport
pyximport.install()

from contextlib import contextmanager

from ..Kernels import Kernel
from . import FlatProjector_cython as cython

class FlatProjector(scipy.sparse.linalg.LinearOperator):

	def __init__(self, kernel):
		if isinstance(kernel, Kernel):

			self.kernel = kernel
			self.shape = kernel.fshape
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
			cython.matvec(v, u, self.kernel.dat, self.kernel.row, self.kernel.col)

		return u
	
	def _rmatvec(self, v):
		v = v.reshape(self.shape[0])
		u = numpy.zeros(self.shape[1], self.dtype)

		with self.in_memory():
			cython.matvec(v, u, self.kernel.dat, self.kernel.col, self.kernel.row)

		return u
