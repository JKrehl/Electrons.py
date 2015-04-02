from __future__ import division, print_function, absolute_import

import numpy
from ....Utilities import FourierTransforms as FT
from .Base import AtomPotentialGenerator

import os.path
__dir__ = os.path.dirname(os.path.abspath(__file__))


class KirklandClass(AtomPotentialGenerator):
	def __init__(self):
		self.coeff = {int(i[0]):i[1:].reshape(4,3)*numpy.array((1e10,1e20,1e-10,1e-20))[:,None] for i in numpy.loadtxt(__dir__+"/parameters/kirkland_coefficients.dat")}

	def form_factors(self, Z, *k):
		qq = reduce(numpy.add.outer,tuple(numpy.require(i)**2 for i in k), 0)[(numpy.s_[:],)*len(k)+(None,)]
		a,b,c,d = self.coeff[Z]
		return numpy.sum(a/(qq+b)+c*numpy.exp(-d*qq),axis=-1)

Kirkland = KirklandClass()
