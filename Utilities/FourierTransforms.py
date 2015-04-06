from __future__ import division

import numpy
import pyfftw

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

def fft(ar, *args, **kwargs):
	if kwargs.has_key('axis'):
		kwargs['axes'] = (kwargs.pop('axis'),)
	return pyfftw.builders.fftn(ar, *args, **kwargs)()

def ifft(ar, *args, **kwargs):
	if kwargs.has_key('axis'):
		kwargs['axes'] = (kwargs.pop('axis'),)
	return pyfftw.builders.ifftn(ar, *args, **kwargs)()

def mfft(ar, *args, **kwargs):
	axes = None
	if kwargs.has_key('axis'):
		axes = (kwargs.pop('axis'),)
		kwargs['axes'] = axes
	if kwargs.has_key('axes'):
		axes = kwargs['axes']
		
	return numpy.fft.fftshift(pyfftw.builders.fftn(numpy.fft.ifftshift(ar,axes=axes), *args, **kwargs)(),axes=axes)

def mifft(ar, *args, **kwargs):
	axes = None
	if kwargs.has_key('axis'):
		axes = (kwargs.pop('axis'),)
		kwargs['axes'] = axes
	elif kwargs.has_key('axes'):
		axes = kwargs['axes']
	else:
		axes = None
	return numpy.fft.fftshift(pyfftw.builders.ifftn(numpy.fft.ifftshift(ar,axes=axes), *args, **kwargs)(),axes=axes)
