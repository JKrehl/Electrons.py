from __future__ import division, print_function, absolute_import

import numpy
import numexpr
from ....Utilities import FourierTransforms as FT
from .Base import AtomPotentialGenerator

import os.path
__dir__ = os.path.dirname(os.path.abspath(__file__))


class KirklandClass(AtomPotentialGenerator):
	def __init__(self):
		self.coeff = {int(i[0]):i[1:].reshape(4,3)*numpy.array((1e10,1e20,1e-10,1e-20))[:,None] for i in numpy.loadtxt(__dir__+"/parameters/kirkland_coefficients.dat")}

	def form_factors(self, Z, *k):
		qq = reduce(numpy.add.outer,tuple(numpy.require(i)**2 for i in k), 0)[(numpy.s_[:],)*len(k)+(None,)]
		
		return numexpr.evaluate('a0/(qq+b0)+a1/(qq+b1)+a2/(qq+b2) + c0*exp(-d0*qq)+c1*exp(-d1*qq)+c2*exp(-d2*qq)',
								local_dict=dict([('qq',qq)]+[("{}{}".format(a,j),self.coeff[Z][i,j]) for i,a in enumerate("abcd") for j in range(3)]))

Kirkland = KirklandClass()
