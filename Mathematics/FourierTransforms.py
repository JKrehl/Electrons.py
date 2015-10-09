from __future__ import division, unicode_literals

import numpy
import pyfftw
import numexpr

def reciprocal_coords(*x):
	if len(x)==1:
		i = x[0]
		return 2*numpy.pi*(i.size-1)/(numpy.ptp(i)*i.size) * ishift((numpy.arange(0,i.size)-i.size//2))
	else:
		return tuple(2*numpy.pi*(i.size-1)/(numpy.ptp(i)*i.size) * ishift((numpy.arange(0,i.size)-i.size//2)) for i in x)
	
def mreciprocal_coords(*x):
	if len(x)==1:
		i = x[0]
		return 2*numpy.pi*(i.size-1)/(numpy.ptp(i)*i.size) * (numpy.arange(0,i.size)-i.size//2)
	else:
		return tuple(2*numpy.pi*(i.size-1)/(numpy.ptp(i)*i.size) * (numpy.arange(0,i.size)-i.size//2) for i in x)
	
def shift(ar, axes=None, axis=None):
	if axis is not None:
		axes = (axis,)
	return numpy.fft.fftshift(ar, axes=axes)

def ishift(ar, axes=None, axis=None):
	if axis is not None:
		axes = (axis,)
	return numpy.fft.ifftshift(ar, axes=axes)

def mwedge(ar, axis=None, axes=None):
	if axis is not None:
		axes = (axis,)
	if axes is None:
		axes = range(ar.ndim)
	
	axes = tuple(i%ar.ndim for i in axes)
	lengths = tuple(ar.shape[i] for i in axes)
	
	ldict = {'f{:d}'.format(i):numpy.exp(1j*2*numpy.pi*(il//2/il*ishift(numpy.arange(-(il//2),(il+1)//2))))[tuple(numpy.s_[:] if j==axes[i] else None for j in range(ar.ndim))] for i, il in enumerate(lengths)}
	ldict['ar'] = ar
	ldict['pi'] = numpy.pi
	
	return numexpr.evaluate("ar*{:s}".format("".join('f{:d}*'.format(i) for i in range(len(axes)))[:-1]), local_dict=ldict)

def miwedge(ar, axis=None, axes=None):
	if axis is not None:
		axes = (axis,)
	if axes is None:
		axes = range(ar.ndim)
	
	axes = tuple(i%ar.ndim for i in axes)
	lengths = tuple(ar.shape[i] for i in axes)
	
	ldict = {'f{:d}'.format(i):numpy.exp(-1j*2*numpy.pi*(il//2)/il*ishift(numpy.arange(-(il//2),(il+1)//2)))[tuple(numpy.s_[:] if j==axes[i] else None for j in range(ar.ndim))] for i, il in enumerate(lengths)}
	ldict['ar'] = ar
	ldict['pi'] = numpy.pi
	
	return numexpr.evaluate("ar*{:s})".format("".join('f{:d}*'.format(i) for i in range(len(axes)))[:-1]), local_dict=ldict)


def mshift(ar, axis=None, axes=None):
	if axis is not None:
		axes = (axis,)
	if axes is None:
		axes = range(ar.ndim)
		
	axes = tuple(i%ar.ndim for i in axes)
	lengths = tuple(ar.shape[i] for i in axes)
	
	ldict = {'fac%d'%i:numpy.linspace(-numpy.pi/2*ar.shape[i], numpy.pi/2*ar.shape[i], ar.shape[i], False)[tuple(numpy.s_[:] if i==j else None for j in range(ar.ndim))] for i in axes}
	ldict.update(j=1j,ar=numpy.fft.fftshift(ar, axes=axes))
	
	return numexpr.evaluate("ar*exp(j*(%s))"%(''.join('fac%d+'%i for i in axes)[:-1]), local_dict=ldict)

def mishift(ar, axes=None, axis=None):
	if axis is not None:
		axes = (axis,)
	if axes is None:
		axes = range(ar.ndim)
			
	axes = tuple(i%ar.ndim for i in axes)
	lengths = tuple(ar.shape[i] for i in axes)
	
	ldict = {'fac%d'%i:numpy.linspace(-numpy.pi/2*ar.shape[i], numpy.pi/2*ar.shape[i], ar.shape[i], False)[tuple(numpy.s_[:] if i==j else None for j in range(ar.ndim))] for i in axes}
	ldict.update(j=1j,ar=numpy.fft.ifftshift(ar, axes=axes))
	
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
