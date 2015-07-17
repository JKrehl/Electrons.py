from __future__ import absolute_import, division

import numpy
import numexpr
from scipy import ndimage
from ....Mathematics import FourierTransforms as FT
from ....Utilities import Physics

from ..Base import IntervalOperator

class FresnelFourier(IntervalOperator):
	def __init__(self, zi, zf, k=None, y=None, x=None, ky=None, kx=None, kk=None):
		self.__dict__.update(dict(zi=zi,zf=zf, wavenumber=k, kk=kk))

		if self.kk is None:
			if ky is None:
				ky = FT.reciprocal_coords(y)
			if kx is None:
				kx = FT.reciprocal_coords(x)
			
			self.kk = numpy.add.outer(ky**2, kx**2)

	@staticmethod
	def prep(k, y=None, x=None, ky=None, kx=None, kk=None):
		if kk is None:
			if ky is None:
				ky = FT.reciprocal_coords(y)
			if kx is None:
				kx = FT.reciprocal_coords(x)
			
			kk = numpy.add.outer(ky**2, kx**2)
		return dict(k=k, kk=kk)
		
	def apply(self, wave):
		return FT.ifft(numexpr.evaluate('wave_f*exp(-1j*pi*dis/wn*kk)', local_dict={'wave_f':FT.fft(wave), 'pi':numpy.pi, 'dis':self.zf-self.zi, 'wn':self.wavenumber, 'kk':self.kk}))

	def split(self, z):
		return FresnelFourier(self.zi, z, self.wavenumber, kk=self.kk), FresnelFourier(z, self.zf, self.wavenumber, kk=self.kk)
