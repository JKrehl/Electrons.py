 
from __future__ import division, print_function, absolute_import

import numpy
from ...Utils import FourierTransforms as FT
from .Base import AtomPotentialGenerator

import os.path
__dir__ = os.path.dirname(os.path.abspath(__file__))

_kds = 1.e10*2

class WeickenmeierKohlClass(AtomPotentialGenerator):
	def __init__(self):
		self.coeff = {int(i[0]):i[1:] for i in numpy.loadtxt(__dir__+"/parameters/weickenmeier_kohl_coefficients.dat")}

	def form_factors(self, Z, *k):
		ss = reduce(numpy.add.outer,tuple((numpy.require(i)/_kds)**2 for i in k), 0)
		A1 = 2.395e-2*Z/(3.+3.*self.coeff[Z][0])
		A4 = self.coeff[Z][0]*A1
		params = zip((A1,A1,A1,A4,A4,A4), self.coeff[Z][1:7])
	
		mss = ss!=0
		re = numpy.empty(ss.shape,type(A1))
		re[mss] = 1.e-10*sum((A*(1-numpy.exp(-B*ss[mss]))/ss[mss] for (A,B) in params))
		re[~mss] = 1.e-10*sum((A*B for (A,B) in params))

		return re
	
WeickenmeierKohl = WeickenmeierKohlClass()
