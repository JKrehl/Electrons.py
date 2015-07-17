from __future__ import absolute_import, division

import numpy
import numexpr
from scipy import ndimage
from ....Mathematics import FourierTransforms as FT
from ....Utilities import Physics

from ..Base import IntervalOperator

class FresnelFourierPadded(IntervalOperator):
	def __init__(self, zi, zf, energy, kx, ky, kk=None):
		if kk is None:
			self.kk = numpy.add.outer(kx**2, ky**2)
		else:
			self.kk = kk

		self.__dict__.update(dict(zi=zi,zf=zf, wavenumber=Physics.wavenumber(energy)))

	def apply(self, wave):
		#return FT.ifft(numexpr.evaluate('wave_f*exp(-1j*pi*dis/wn*kk)', local_dict={'wave_f':FT.fft(wave), 'pi':numpy.pi, 'dis':self.zf-self.zi, 'wn':self.wavenumber, 'kk':self.kk}))
		pad = tuple(i//2 for i in self.kk.shape)
		pshape = tuple(i+2*j for i,j in zip(self.kk.shape, pad))
		scl = tuple(i/j for i,j in zip(self.kk.shape, pshape))
		depad = tuple(slice(i,-i) for i in pad)
		win = numpy.multiply.reduce(numpy.ix_(*tuple(numpy.hamming(i) for i in wave.shape)))
		return FT.ifft(numexpr.evaluate('wave_f*exp(-1j*pi*dis/wn*kk)', local_dict={'wave_f':FT.fft(numpy.pad(wave*win,pad,mode='constant')), 'pi':numpy.pi, 'dis':self.zf-self.zi, 'wn':self.wavenumber,
																					'kk':ndimage.interpolation.affine_transform(self.kk,scl,output_shape=pshape)}))[depad]/win

	def split(self, z):
		return FresnelFourier(self.zi, z, self.wavenumber, kk=self.kk), FresnelFourier(z, self.zf, self.wavenumber, kk=self.kk)
