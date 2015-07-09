from __future__ import absolute_import, division

import numpy
import numexpr
import scipy.ndimage
import scipy.optimize

from ....Utilities import FourierTransforms as FT, Physics

from ..Base import IntervalOperator

def laplace_kernel(r):
	y = numpy.arange(-r,r+1)
	x = numpy.arange(-r,r+1)
	
	sy = numpy.arange(r+1)
	sx = numpy.arange(1,r+1)
	
	sy, sx = (i.flatten() for i in numpy.meshgrid(sy,sx, indexing='ij'))
	sr = numpy.sqrt(sy**2+sx**2)
	sth = numpy.arctan2(sy, sx)
	
	kr = numpy.linspace(0,numpy.pi,16*(r+2))[1:]
	th = numpy.linspace(0,numpy.pi/2, 64)
	kr, th = numpy.meshgrid(kr, th, indexing='ij')
	
	rgains = numexpr.evaluate("1/sr**2*(4-2*cos(kr*sr*cos(th+sth))-2*cos(kr*sr*sin(th+sth)))/kr**2", local_dict=dict(kr=kr[None,:,:], th=th[None,:,:], sr=sr[:,None,None], sth=sth[:,None,None]))
	
	def rgain(p):
		p = numpy.concatenate(([1-p.sum()], p.flatten()))
		return numexpr.evaluate("sum(p*rgains, 0)", local_dict=dict(rgains=rgains, p=p[:,None,None]))

	refgain = numpy.sinc(1/(r+1/2)*kr/numpy.pi)
	
	def tomin(p):
		rg = rgain(p)
		#a = (rg.std(-1)**2).sum()
		#b = (kr*rg).mean()**2
		
		#penalty = numpy.count_nonzero((rg>1)|(rg<0))
		#penalty = numpy.count_nonzero(numpy.diff(rg, axis=0)>0)
		return (rg-refgain).ptp(-1).sum()
	
	fit = scipy.optimize.minimize(tomin, numpy.zeros(sy.size-1), method='Powell')

    
	p = numpy.concatenate(([1-fit.x.sum()], fit.x.flatten()))

	kernel = numpy.zeros(y.shape+x.shape)
	for i in xrange(len(p)):
		kernel[(r,r-sy[i],r-sx[i],r+sy[i],r+sx[i]),(r,r+sx[i],r-sy[i],r-sx[i],r+sy[i])] += p[i]/sr[i]**2*numpy.array((-4,1,1,1,1))
		
	return kernel

__kernels = dict()
def get_kernel(r):
	if not __kernels.has_key(r):
		__kernels[r] = laplace_kernel(r)
		
	return __kernels[r]
	
class FresnelRealspace(IntervalOperator):
	def __init__(self, zi, zf, k, y=None, x=None, r=1, kernel=None):
		self.__dict__.update(dict(zi=zi,zf=zf, k=k, kernel=kernel))
		if self.kernel is None:
			self.kernel = get_kernel(r)/(((y[1]-y[0])*(x[1]-x[0])))

	def apply(self, wave):
		laplace = scipy.ndimage.convolve(wave.real, self.kernel, mode='constant', cval=1) + 1j*scipy.ndimage.convolve(wave.imag, self.kernel, mode='constant', cval=0)
		return numexpr.evaluate('wave*exp(-j*dis/(2*k)*laplace)', local_dict=dict(wave=wave, j=1j, pi=numpy.pi, dis=self.zf-self.zi, k=self.k, laplace=laplace))

	def split(self, z):
		return FresnelRealspace(self.zi, z, self.k, kernel=self.kernel), FresnelRealspace(z, self.zf, self.k, kernel=self.kernel)
