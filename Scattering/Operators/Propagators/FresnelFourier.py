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
from scipy import ndimage
from ....Mathematics import FourierTransforms as FT
from ....Utilities import Physics

from ..Operator import IntervalOperator

class FresnelFourier(IntervalOperator):
	def __init__(self, zi, zf,
	             kk = None, ky = None, kx = None,
	             k = None, energy = None,
	             y = None, x = None,
	             factory = False):
		super().__init__(zi, zf)

		if kk is None:
			if ky is None: ky = FT.reciprocal_coords(y)
			if kx is None: kx = FT.reciprocal_coords(x)
			self.kk = numpy.add.outer(ky**2, kx**2)
		else:
			self.kk = kk

		if k is None: self.k = Physics.wavenumber(energy)
		else: self.k = k

	def derive(self, zi, zf, **kwargs):
		args = dict(k=self.k, kk=self.kk)
		args.update(kwargs)

		return self.__class__(zi, zf, **args)

	def apply(self, wave):
		return FT.ifft(numexpr.evaluate('wave_f*exp(-1j*dis/(2*wn)*kk)', local_dict={'wave_f':FT.fft(wave), 'pi':numpy.pi, 'dis':self.zf-self.zi, 'wn':self.k, 'kk':self.kk}))

	def split(self, z):
		return self.derive(self.zi, z), self.derive(z, self.zf)

