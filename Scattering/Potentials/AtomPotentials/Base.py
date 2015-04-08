from __future__ import absolute_import, division

import numpy

from ....Utilities.Physics import bohrr, echarge, interaction_const
from ....Utilities import FourierTransforms as FT

class AtomPotentialGenerator:
	def __init__(self):
		pass

	def form_factors(self, Z, *k):
		raise NotImplementedError

	def real_space_ff(self, Z, *x):
		raise NotImplementedError

	def potential_from_ff(self, ff, *x):
		area = numpy.multiply.reduce(tuple(numpy.ptp(i) for i in x))
		size = numpy.multiply.reduce(tuple(i.size for i in x))
		return abs(FT.mifft(ff))*2*numpy.pi*bohrr*echarge/(area/size)

	def real_space_potential(self, Z, *x):
		area = numpy.multiply.reduce(tuple(numpy.ptp(i) for i in x))
		size = numpy.multiply.reduce(tuple(i.size for i in x))
		
		ff = self.real_space_ff(Z, *x)
		
		return abs(ff)*2*numpy.pi*bohrr*echarge/(area/size)
	
	def potential(self, Z, *x):
		k = FT.reciprocal_coords(*x)
		return self.potential_from_ff(self.form_factors(Z, *k), *x)

	def phaseshift(self, Z, energy,  *x):
		return numpy.exp(1j*interaction_const(energy)*self.potential(Z, *x))

	def phaseshift_f(self, Z, energy, *x):
		return FT.fft(self.phaseshift(Z, energy, *x))

