from __future__ import absolute_import, division

import numpy
import numexpr
from scipy import ndimage
from ...Utils import FourierTransforms as FT, Physics

from ..Base import IntervalOperator

class MagnusExpansion(IntervalOperator):
	def __init__(self, zi, zf, energy, order=1, kx, ky, kk=None):
		self.order = order
		
		if kk is None:
			self.kk = numpy.add.outer(kx**2, ky**2)
		else:
			self.kk = kk

		self.__dict__.update(dict(zi=zi,zf=zf, wavenumber=Physics.wavenumber(energy)))

	def apply(self, wave):
		if self.order == 1:
			return FT.ifft(numexpr.evaluate('wave_f*exp(1-1j*pi*dis/wn*kk)', local_dict={'wave_f':FT.fft(wave), 'pi':numpy.pi, 'dis':self.zf-self.zi, 'wn':self.wavenumber, 'kk':self.kk}))
		else:
			raise NotImplemented

	def split(self, z):
		return FresnelFourier(self.zi, z, self.wavenumber, kk=self.kk), FresnelFourier(z, self.zf, self.wavenumber, kk=self.kk)
