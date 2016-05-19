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

from .AtomPotential import AtomPotential

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
