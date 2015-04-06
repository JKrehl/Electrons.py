 
from __future__ import division, print_function, absolute_import

import numpy
import numexpr
from ....Utilities import FourierTransforms as FT
from .Base import AtomPotentialGenerator

import os.path
__dir__ = os.path.dirname(os.path.abspath(__file__))

_kds = 1.e10*2

class WeickenmeierKohlClass(AtomPotentialGenerator):
	def __init__(self):
		self.coeff = {int(i[0]):i[1:] for i in numpy.loadtxt(__dir__+"/parameters/weickenmeier_kohl_coefficients.dat")}

	def form_factors(self, Z, *k):
		ss = reduce(numpy.add.outer,tuple((numpy.require(i)/_kds)**2 for i in k), 0)
		A0 = 2.395e-2*Z/(3.+3.*self.coeff[Z][0])
		A3 = self.coeff[Z][0]*A0
		B = self.coeff[Z][1:7]
	
		mss = ss!=0
		re = numpy.empty(ss.shape,type(A0))
		re[mss] = numexpr.evaluate("1e-10*(A0*(1-exp(-B0*ss))/ss+A0*(1-exp(-B1*ss))/ss+A0*(1-exp(-B2*ss))/ss+A3*(1-exp(-B3*ss))/ss+A3*(1-exp(-B4*ss))/ss+A3*(1-exp(-B5*ss))/ss)",
								   local_dict=dict(ss=ss[mss], A0=A0,A3=A3,B0=B[0],B1=B[1],B2=B[2],B3=B[3],B4=B[4],B5=B[5]))
		re[~mss] = 1e-10*(A0*B[0]+A0*B[1]+A0*B[2]+A3*B[3]+A3*B[4]+A3*B[5])

		return re
	
WeickenmeierKohl = WeickenmeierKohlClass()
