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
import numexpr

from .AtomPotential import AtomPotential

import functools
import os.path

__dir__ = os.path.dirname(os.path.abspath(__file__))
coeff = {int(i[0]):i[1:] for i in numpy.loadtxt(__dir__+"/parameters/peng_dudarev_coefficients.dat")}

_kds = 1.e10*4*numpy.pi

class PengDudarev(AtomPotential):
	def __init__(self):
		pass

	@classmethod
	def form_factors_k(cls, Z, *k):
		ss = functools.reduce(numpy.add.outer,tuple((numpy.require(i)/_kds)**2 for i in k), 0)
		a0,a1,a2,a3,a4,b0,b1,b2,b3,b4 = coeff[Z]
		return numexpr.evaluate("1e-10*(a0*exp(-b0*ss)+a1*exp(-b1*ss)+a2*exp(-b2*ss)+a3*exp(-b3*ss)+a4*exp(-b4*ss))",local_dict=dict(ss=ss,a0=a0,a1=a1,a2=a2,a3=a3,a4=a4,b0=b0,b1=b1,b2=b2,b3=b3,b4=b4))
