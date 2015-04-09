from __future__ import division, print_function, absolute_import

import numpy
import numexpr
from ....Utilities import FourierTransforms as FT
from .Base import AtomPotentialGenerator

import os.path
__dir__ = os.path.dirname(os.path.abspath(__file__))

_kds = 1.e10*2

class PengDudarevClass(AtomPotentialGenerator):
	def __init__(self):
		self.coeff = {int(i[0]):i[1:] for i in numpy.loadtxt(__dir__+"/parameters/peng_dudarev_coefficients.dat")}

	def form_factors_k(self, Z, *k):
		ss = reduce(numpy.add.outer,tuple((numpy.require(i)/_kds)**2 for i in k), 0)
		a0,a1,a2,a3,a4,b0,b1,b2,b3,b4 = self.coeff[Z]
		return numexpr.evaluate("1e-10*(a0*exp(-b0*ss)+a1*exp(-b1*ss)+a2*exp(-b2*ss)+a3*exp(-b3*ss)+a4*exp(-b4*ss))",local_dict=dict(ss=ss,a0=a0,a1=a1,a2=a2,a3=a3,a4=a4,b0=b0,b1=b1,b2=b2,b3=b3,b4=b4))

PengDudarev = PengDudarevClass()
