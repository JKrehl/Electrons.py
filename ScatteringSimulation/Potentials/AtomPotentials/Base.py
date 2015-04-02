from __future__ import absolute_import, division

import numpy

from ...Utils.Physics import bohrr, echarge
from ...Utils import FourierTransforms as FT, Physics

class AtomPotentialGenerator:
	def __init__(self):
		pass

	def form_factors(self, Z, *k):
		raise NotImplemented

	def potential_from_ff(self, ff, *x):
		area = numpy.multiply.reduce(tuple(numpy.ptp(i) for i in x))
		size = numpy.multiply.reduce(tuple(i.size for i in x))
		return abs(FT.mifft(ff))*2*numpy.pi*bohrr*echarge/(area/size)
		
	def potential(self, Z, *x):
		k = FT.reciprocal_coords(*x)
		return self.potential_from_ff(self.form_factors(Z, *k), *x)

	def phaseshift(self, Z, energy,  *x):
		return numpy.exp(1j*Physics.interaction_const(energy)*self.potential(Z, *x))

	def phaseshift_f(self, Z, energy, *x):
		return FT.fft(self.phaseshift(Z, energy, *x))

