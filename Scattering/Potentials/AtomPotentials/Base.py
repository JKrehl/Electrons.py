from __future__ import absolute_import, division

import numpy

from ....Utilities.Physics import bohrr, echarge, interaction_const
from ....Utilities import FourierTransforms as FT

class AtomPotentialGenerator:
	def __init__(self):
		pass

	def form_factors_k(self, Z, *k):
		raise NotImplementedError

	def form_factors_r(self, Z, *x):
		raise NotImplementedError

	def potential_from_ff(self, ff):
		return ff*2*numpy.pi*bohrr*echarge

	def potential_r(self, Z, *x):
		return self.potential_from_ff(self.form_factors_r(Z, *x))

	def potential_k(self, Z, *x, **kwargs):
		if kwargs.has_key('msk'):
			msk = kwargs['msk']
		else:
			msk = None
			
		k = FT.reciprocal_coords(*x)
		area = numpy.multiply.reduce(tuple(numpy.ptp(i) for i in x))
		size = numpy.multiply.reduce(tuple(i.size for i in x))

		if msk is not None:
			return self.potential_from_ff(abs(FT.mifft(msk*self.form_factors_k(Z, *k)))/(area/size))
		else:
			return self.potential_from_ff(abs(FT.mifft(self.form_factors_k(Z, *k)))/(area/size))

	def potential(self, Z, *x):
		return self.potential_k(Z, *x)
	
	def phaseshift(self, Z, energy,  *x):
		return interaction_const(energy)*self.potential(Z, *x)

	def phaseshift_f(self, Z, energy, *x):
		return FT.fft(self.phaseshift(Z, energy, *x))

