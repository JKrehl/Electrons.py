from __future__ import division, unicode_literals

import numpy
import pyfftw
import numexpr

def reciprocal_coords(*x):
	if len(x)==1:
		i = x[0]
		return 2*numpy.pi*i.size/(numpy.ptp(i)*(i.size+1)) * ishift((numpy.arange(0,i.size)-i.size//2))
	else:
		return tuple(2*numpy.pi*i.size/(numpy.ptp(i)*(i.size+1)) * ishift((numpy.arange(0,i.size)-i.size//2)) for i in x)
	
def mreciprocal_coords(*x):
	if len(x)==1:
		i = x[0]
		return 2*numpy.pi*i.size/(numpy.ptp(i)*(i.size+1)) * (numpy.arange(0,i.size)-i.size//2)
	else:
		return tuple(2*numpy.pi*i.size/(numpy.ptp(i)*(i.size+1)) * (numpy.arange(0,i.size)-i.size//2) for i in x)
	
def shift(ar, axes=None, axis=None):
	if axis is not None:
		axes = (axis,)
	return numpy.fft.fftshift(ar, axes=axes)

def ishift(ar, axes=None, axis=None):
	if axis is not None:
		axes = (axis,)
	return numpy.fft.ifftshift(ar, axes=axes)

def mshift(ar, axis=None, axes=None):
	if axis is not None:
		axes = (axis,)
	if axes is None:
		axes = range(ar.ndim)
		
	axes = tuple(i%ar.ndim for i in axes)
	lengths = tuple(ar.shape[i] for i in axes)
	
	ldict = {'fac%d'%i:numpy.linspace(-numpy.pi/2*ar.shape[i], numpy.pi/2*ar.shape[i], ar.shape[i], False)[tuple(numpy.s_[:] if i==j else None for j in xrange(ar.ndim))] for i in axes}
	ldict.update(j=1j,ar=numpy.fft.fftshift(ar, axes=axes))
	
	return numexpr.evaluate("ar*exp(j*(%s))"%(''.join('fac%d+'%i for i in axes)[:-1]), local_dict=ldict)

def mishift(ar, axes=None, axis=None):
	if axis is not None:
		axes = (axis,)
	if axes is None:
		axes = range(ar.ndim)
			
	axes = tuple(i%ar.ndim for i in axes)
	lengths = tuple(ar.shape[i] for i in axes)
	
	ldict = {'fac%d'%i:numpy.linspace(-numpy.pi/2*ar.shape[i], numpy.pi/2*ar.shape[i], ar.shape[i], False)[tuple(numpy.s_[:] if i==j else None for j in xrange(ar.ndim))] for i in axes}
	ldict.update(j=1j,ar=numpy.fft.fftishift(ar, axes=axes))
	
	return numexpr.evaluate("ar*exp(-j*(%s))"%(''.join('fac%d+'%i for i in axes)[:-1]), local_dict=ldict)
def fft(ar, *args, **kwargs):
	if 'axis' in kwargs:
		kwargs['axes'] = (kwargs.pop('axis'),)
	return pyfftw.builders.fftn(ar, *args, **kwargs)()

def ifft(ar, *args, **kwargs):
	if 'axis' in kwargs:
		kwargs['axes'] = (kwargs.pop('axis'),)
	return pyfftw.builders.ifftn(ar, *args, **kwargs)()

def mfft(ar, *args, **kwargs):
	axes = None
	if 'axis' in kwargs:
		axes = (kwargs.pop('axis'),)
		kwargs['axes'] = axes
	if 'axes' in kwargs:
		axes = kwargs['axes']
		
	return numpy.fft.fftshift(pyfftw.builders.fftn(numpy.fft.ifftshift(ar,axes=axes), *args, **kwargs)(),axes=axes)

def mifft(ar, *args, **kwargs):
	axes = None
	if 'axis' in kwargs:
		axes = (kwargs.pop('axis'),)
		kwargs['axes'] = axes
	elif 'axes' in kwargs:
		axes = kwargs['axes']
	else:
		axes = None
	return numpy.fft.fftshift(pyfftw.builders.ifftn(numpy.fft.ifftshift(ar,axes=axes), *args, **kwargs)(),axes=axes)
