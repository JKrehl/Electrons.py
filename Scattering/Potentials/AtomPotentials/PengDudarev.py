from __future__ import division, print_function, absolute_import

import numpy
from ....Utilities import FourierTransforms as FT
from .Base import AtomPotentialGenerator

import os.path
__dir__ = os.path.dirname(os.path.abspath(__file__))

_kds = 1.e10*2

class PengDudarevClass(AtomPotentialGenerator):
	def __init__(self):
		self.coeff = {int(i[0]):i[1:] for i in numpy.loadtxt(__dir__+"/parameters/peng_dudarev_coefficients.dat")}

	def form_factors(self, Z, *k):
		ss = reduce(numpy.add.outer,tuple((numpy.require(i)/_kds)**2 for i in k), 0)
		return 1e-10*sum((A*numpy.exp(-B*ss) for (A,B) in zip(self.coeff[Z][0:5],self.coeff[Z][5:10])))

PengDudarev = PengDudarevClass()
