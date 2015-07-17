import numpy

from ....Mathematics import FourierTransforms as FT

from .Base import AtomPotential

import functools
import os.path

__dir__ = os.path.dirname(os.path.abspath(__file__))
coeff = {int(i[0]):i[1:] for i in numpy.loadtxt(__dir__+"/parameters/weickenmeier_kohl_fscatt_coefficients.dat")}

_kds = 1.e10*4*numpy.pi

class WeickenmeierKohlFSCATT(AtomPotential):
	def __init__(self):
		pass

	@classmethod
	def form_factors_k(cls, Z, *k):
		ss = functools.reduce(numpy.add.outer,tuple((numpy.require(i)/_kds)**2 for i in k), 0)
		mss = ss!=0
		
		re = numpy.empty_like(ss, type(coeff[Z][0]))
		re[mss] = (lambda ss:1.e-10*sum(numpy.where(B*ss>1/(2*numpy.pi), A/ss*numpy.where(B*ss>20/(2*numpy.pi), 1, (1-numpy.exp(-B*ss))), A*B*(1-.5*B*ss)) for (A,B) in zip(coeff[Z][:4], coeff[Z][4:8])))(ss[mss])
		re[~mss] = (lambda ss:1.e-10*sum(A*B for (A,B) in zip(coeff[Z][:4], coeff[Z][4:8])))(ss[~mss])
		return re
