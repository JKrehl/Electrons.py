#!/usr/bin/env python
"""
Copyright (c) 2015 Jonas Krehl <Jonas.Krehl@triebenberg.de>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

import numpy
import scipy.sparse
import pyximport
pyximport.install()

from contextlib import contextmanager

from ..Kernels import Kernel
from .FlatProjector import FlatProjector
from . import StackedProjector_cython as cython

class StackedProjector(scipy.sparse.linalg.LinearOperator):
	def __init__(self, kernel, z, threads=0):
		self.z = z
		self.threads = threads

		if isinstance(kernel, Kernel):

			self.kernel = kernel
			self.shape = tuple(i*self.z.size for i in kernel.fshape)

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
			cython.matvec(v,u, self.kernel.dat, self.kernel.col, self.kernel.row, self.z.size, self.kernel.fshape[1], self.kernel.fshape[0], self.threads)

		return u
	
	def _rmatvec(self, v):
		v = v.reshape(self.shape[0])
		
		u = numpy.zeros(self.shape[1], self.dtype)

		if numpy.issubdtype(self.kernel.dat.dtype, numpy.complex):
			numpy.conj(v,v)

		with self.in_memory():
			cython.matvec(v,u, self.kernel.dat, self.kernel.row, self.kernel.col, self.z.size, self.kernel.fshape[0], self.kernel.fshape[1], self.threads)

		if numpy.issubdtype(self.kernel.dat.dtype, numpy.complex):
			numpy.conj(u,u)
			numpy.conj(v,v)

		return u