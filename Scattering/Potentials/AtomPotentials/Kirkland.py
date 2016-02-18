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
from ....Mathematics import FourierTransforms as FT

from .Base import AtomPotential

import functools
import os.path

__dir__ = os.path.dirname(os.path.abspath(__file__))
coeff = {int(i[0]):i[1:].reshape(4,3)*numpy.array((1e10,1e20,1e-10,1e-20))[:,None] for i in numpy.loadtxt(__dir__+"/parameters/kirkland_coefficients.dat")}

class Kirkland(AtomPotential):
	def __init__(self):
		pass

	@classmethod
	def form_factors_k(cls, Z, *k):
		qq = functools.reduce(numpy.add.outer,tuple((numpy.require(i)/(2*numpy.pi))**2 for i in k), 0)
		a,b,c,d = coeff[Z]
		return numexpr.evaluate('a0/(qq+b0)+a1/(qq+b1)+a2/(qq+b2) + c0*exp(-d0*qq)+c1*exp(-d1*qq)+c2*exp(-d2*qq)',
								local_dict=dict(qq=qq, a0=a[0],a1=a[1],a2=a[2], b0=b[0],b1=b[1],b2=b[2], c0=c[0],c1=c[1],c2=c[2], d0=d[0],d1=d[1],d2=d[2]))

	@classmethod
	def form_factors_r(cls, Z, *x):
		#the scaling factor 2*pi compensates for the Kirkland factors not using proper angular frequencies
		r = 2*numpy.pi*numpy.sqrt(functools.reduce(numpy.add.outer,tuple(numpy.require(i)**2 for i in x), 0))

		return numexpr.evaluate('2*pi**2/r*(a0*exp(-sqrt(b0)*r)+a1*exp(-sqrt(b1)*r)+a2*exp(-sqrt(b2)*r))+sqrt(pi)*(c0/sqrt(d0)*exp(-r**2/(4*d0))+c1/sqrt(d1)*exp(-r**2/(4*d1))+c2/sqrt(d2)*exp(-r**2/(4*d2)))',
								local_dict=dict([('r',r),('pi',numpy.pi)]+[("{}{}".format(a,j),coeff[Z][i,j]) for i,a in enumerate("abcd") for j in range(3)]))
