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

from ....Utilities.Physics import bohrr, echarge, interaction_const
from ....Mathematics import FourierTransforms as FT

class AtomPotential:
	def __init__(self):
		pass

	@classmethod
	def form_factors_k(cls, Z, *k):
		raise NotImplementedError

	@classmethod
	def form_factors_r(cls, Z, *x):
		raise NotImplementedError

	@classmethod
	def potential_r(cls, Z, *x):
		return 2*numpy.pi*bohrr*echarge*cls.form_factors_r(Z, *x)

	@classmethod
	def potential_k(cls, Z, *x):
		k = FT.reciprocal_coords(*x)
		
		scaling = numpy.multiply.reduce(tuple(i[1]-i[0] for i in x))
		
		return 2*numpy.pi*bohrr*echarge*FT.ifft(FT.miwedge(cls.form_factors_k(Z, *k))).real/scaling

	@classmethod
	def proj_potential(cls, Z, y, x, z=None):
		if z is None:
			return cls.potential_k(Z, y, x)
		else:
			return cls.potential_r(Z, z, y, x).sum(0)*(z[1]-z[0])

	@classmethod
	def phaseshift(cls, Z, energy, y, x, z=None):
		return interaction_const(energy)*cls.proj_potential(Z, y, x, z)

	@classmethod
	def phaseshift_f(cls, Z, energy, y, x, z=None):
		return FT.fft(cls.phaseshift(Z, energy, y, x, z))

	@classmethod
	def cis_phaseshift_f(cls, Z, energy, y, x, z=None):
		return FT.fft(numpy.exp(1j*cls.phaseshift(Z, energy, y, x, z)))