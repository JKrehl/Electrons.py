from __future__ import absolute_import, division

import numpy
import numexpr
from scipy import ndimage
from ....Mathematics import FourierTransforms as FT

from ..Base import IntervalOperator

class FresnelFourier(IntervalOperator):
	def __init__(self, zi, zf, k, kk=None, ky=None, kx=None, y=None, x=None):
		self.__dict__.update(dict(zi=zi,zf=zf, k=k, kk=kk))

		if self.kk is None:
			if ky is None:
				ky = FT.reciprocal_coords(y)
			if kx is None:
				kx = FT.reciprocal_coords(x)
			
			self.kk = numpy.add.outer(ky**2, kx**2)
		
	def apply(self, wave):
		return FT.ifft(numexpr.evaluate('wave_f*exp(-1j*pi*dis/wn*kk)', local_dict={'wave_f':FT.fft(wave), 'pi':numpy.pi, 'dis':self.zf-self.zi, 'wn':self.k, 'kk':self.kk}))

	def split(self, z):
		return FresnelFourier(self.zi, z, self.k, self.kk), FresnelFourier(z, self.zf, self.k, self.kk)

	@classmethod
	def inherit(cls, parent, zi, zf, **kwargs):
		k = parent.k
		args = {}

		if hasattr(parent, 'kk'):
			args.update(dict(kk = parent.kk))
		elif hasattr(parent, 'ky') and hasattr(parent, 'kx'):
			args.update(dict(ky = parent.ky, kx = parent.kx))
		elif hasattr(parent, 'y') and hasattr(parent, 'x'):
			args.update(dict(y = parent.y, x = parent.x))

		args.update(kwargs)
			
		return cls(zi, zf, k, **args)
