 
from __future__ import division, print_function, absolute_import

import numpy
from ....Utilities import FourierTransforms as FT
from .Base import AtomPotentialGenerator

import os.path
__dir__ = os.path.dirname(os.path.abspath(__file__))

_kds = 1.e10*2

class WeickenmeierKohlFSCATTClass(AtomPotentialGenerator):
	def __init__(self):
		self.coeff = {int(i[0]):i[1:] for i in numpy.loadtxt(__dir__+"/parameters/weickenmeier_kohl_fscatt_coefficients.dat")}

	def form_factors_k(self, Z, *k):
		ss = reduce(numpy.add.outer,tuple((numpy.require(i)/_kds)**2 for i in k), 0)
		mss = ss!=0
		
		re = numpy.empty_like(ss, type(self.coeff[Z][0]))
		re[mss] = (lambda ss:1.e-10*sum(numpy.where(B*ss>1, A/ss*numpy.where(B*ss>20, 1, (1-numpy.exp(-B*ss))), A*B*(1-.5*B*ss)) for (A,B) in zip(self.coeff[Z][:4], self.coeff[Z][4:8])))(ss[mss])
		re[~mss] = (lambda ss:1.e-10*sum(A*B for (A,B) in zip(self.coeff[Z][:4], self.coeff[Z][4:8])))(ss[~mss])
		return re
	
WeickenmeierKohlFSCATT = WeickenmeierKohlFSCATTClass()
