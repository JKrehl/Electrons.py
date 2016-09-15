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

from ...Utilities import Progress

from .Operator import IntervalOperator, PlaneOperator

class OperatorChain(numpy.ndarray):
	def __new__(cls, shape=0, array=None, zi=None, zf=None):
		if array is not None:
			return numpy.asanyarray(array, dtype=dict(names=['zi','zf', 'operator'], formats=[numpy.float,numpy.float,object]))
		else:
			return numpy.ndarray.__new__(cls, shape, dtype=dict(names=['zi','zf', 'operator'], formats=[numpy.float,numpy.float,object]))

	def __init__(self, shape=0, array=None, zi=None, zf=None):
		self.zi, self.zf = zi, zf

	def __array_finalize__(self, obj):
		if obj is None: return
		self.zi = getattr(obj, 'zi', None)
		self.zf = getattr(obj, 'zf', None)

	def insert(self, position, operator):
		self.resize(self.size+1, refcheck=False)
		self[position+1:] = self[position:-1]

		if isinstance(operator, PlaneOperator):
			self[position] = (operator.z, operator.z, operator)
		elif isinstance(operator, IntervalOperator):
			self[position] = (operator.zi, operator.zf, operator)
		else:
			raise NotImplementedError("Operators of type %s cannot be integrated into operator chain."%str(type(operator)))

	def append(self, operator):
		self.insert(self.size, operator)

	def prepend(self, operator):
		self.insert(0, operator)

	def zinsert(self, operator, z=None):
		if z is None: z = operator.z

		arg = numpy.nonzero((self['zi'] == z)&(self['zi'] < self['zf']))[0]
		if len(arg) != 0:
			self.insert(arg[0], operator)
			return True

		arg = tuple(numpy.nonzero((self['zi'] < z)&(self['zf'] > z)))[0]
		if len(arg) != 0:
			ops = self['operator'][arg[0]].split(z)
			self[arg[0]] = (ops[1].zi, ops[1].zf, ops[1])
			self.insert(arg[0], operator)
			self.insert(arg[0], ops[0])
			return True

		raise NotImplementedError

	def sortz(self):
		self[...] = self[numpy.argsort(self['zf'], kind='mergesort')]
		self[...] = self[numpy.argsort(self['zi'], kind='mergesort')]

	def iszcontinuous(self):
		return len(self.get_gaps())==0

	def get_gaps(self, indices=False):
		if self.size == 0:
			return []

		zfs = self['zf'][:-1]
		zis = self['zi'][1:]
		if self.zf is not None:
			zfs = numpy.hstack((zfs, self['zf'][-1]))
			zis = numpy.hstack((zis, self.zf))
		if self.zi is not None:
			zfs = numpy.hstack((self.zi, zfs))
			zis = numpy.hstack((self['zi'][0], zis))

		gaps = numpy.vstack((zfs, zis)).T
		sel = numpy.diff(gaps).flatten()!=0
		if indices:
			return numpy.vstack((numpy.arange(gaps.shape[0])[sel], gaps[sel, :].T)).T
		else:
			return gaps[numpy.diff(gaps).flatten()!=0,:]

	def apply(self, wave, progress=False):
		self.impose_zorder()

		if progress:
			for op in Progress(self['operator'], self.size):
				wave = op.apply(wave)
		else:
			for op in self['operator']:
				wave = op.apply(wave)

		return wave
