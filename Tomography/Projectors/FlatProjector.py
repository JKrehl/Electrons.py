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
