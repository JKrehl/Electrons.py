from __future__ import absolute_import, print_function, division

import numpy

from ...Utilities import Progress

from .Base import IntervalOperator, PlaneOperator

class OperatorChain(numpy.ndarray):
	def __new__(cls, *args, **kwargs):
		return numpy.ndarray.__new__(cls, 0, dict(names=['zi','zf', 'operator'], formats=[numpy.float,numpy.float,object])).copy()
	def __array_finalize__(self, obj):
		pass
	def __init__(self, zi=None, zf=None):
		self.__dict__.update(dict(zi=zi,zf=zf))

	def impose_zorder(self):
		self[...] = self[numpy.argsort(self['zf'], kind='mergesort')]
		self[...] = self[numpy.argsort(self['zi'], kind='mergesort')]

	def get_gaps(self):
		self.impose_zorder()
		if self.size == 0:
			if self.zi != self.zf:
				return [(self.zi, self.zf)]
			else:
				return []
		else:
			return [(zi,zf) for zi,zf in zip([self.zi]+list(self['zf']), list(self['zi'])+[self.zf]) if zi is not None and zf is not None and not zf==zi]

	def append(self, operator, zi=None, zf=None):			
		if zi is None:
			if hasattr(operator, 'zi'):
				zi = operator.zi
			elif hasattr(operator, 'z') and operator.z is not None:
				zi = operator.z
			else:
				zi = numpy.amax(self['zf'])

		if zf is None:
			if hasattr(operator, 'zf'):
				zf = operator.zf
			else:
				zf = zi

		assert zf is not None
				
		self.resize(self.size+1, refcheck=False)
		
		self[-1] = (zi, zf, operator)

	def apply(self, wave, progress=False):
		self.impose_zorder()

		if progress:
			for op in Progress(self['operator'], self.size):
				wave = op.apply(wave)
		else:
			for op in self['operator']:
				wave = op.apply(wave)

		return wave
