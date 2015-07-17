import numpy
import scipy.ndimage
import scipy.optimize
import scipy.special

import numexpr
from matplotlib.pyplot import *

from . import FourierTransforms as FT

def dderivatecoeff(r):
	x = numpy.arange(-r,r+1, dtype=numpy.float)

	delta = numpy.zeros((3,x.size,x.size))
	delta[0,0,0] = 1.
	c1 = 1.
	for n in range(1, x.size):
		c2 = 1.
		for nu in range(0, n):
			c3 = x[n]-x[nu]
			c2 *= c3
			for m in range(0, min(n+1, delta.shape[0])):
				delta[m,n,nu] = (x[n]*delta[m,n-1,nu]-m*delta[m-1,n-1,nu])/c3
		for m in range(0, min(n+1, delta.shape[0])):
			delta[m,n,n] = c1/c2*(m*delta[m-1,n-1,n-1]-x[n-1]*delta[m,n-1,n-1])
		c1 = c2

	return delta[2,-1,r+1:]

def generate_laplace_kernel(yr, dy=1, dx=None, xr=None):
	if xr is None: xr=yr
	if dx is None: dx=dy
	
	y = numpy.arange(-yr, yr+1)
	x = numpy.arange(-xr, xr+1)
	
	sy = numpy.concatenate((numpy.arange(1, yr+1), numpy.zeros(xr)))
	sx = numpy.concatenate((numpy.zeros(yr), numpy.arange(1, xr+1)))
	sr = numpy.sqrt((dy*sy)**2+(dx*sx)**2)
	sth = numpy.arctan2(sy, sx)
	
	kr = numpy.linspace(0,numpy.pi/min(dy,dx),64)[1:]
	th = numpy.linspace(-numpy.pi,numpy.pi, 64)
	kr, th = numpy.meshgrid(kr, th, indexing='ij')
	
	rgains = numexpr.evaluate("1/sr**2*(2-2*cos(kr*sr*cos(th-sth)))/kr**2", local_dict=dict(kr=kr[None,:,:], th=th[None,:,:], sr=sr[:,None,None], sth=sth[:,None,None]))
	
	def rgain(p):
		return numexpr.evaluate("sum(p*rgains, 0)", local_dict=dict(rgains=rgains, p=p[:,None,None]))
	
	pref = numpy.concatenate((dderivatecoeff(yr), dderivatecoeff(xr)))
	ref = numpy.tile(rgain(pref).min(1), (th.shape[1],1)).T
	
	asy, asx = (i.flatten() for i in numpy.meshgrid(numpy.concatenate((numpy.arange(-yr,0), numpy.arange(1,yr+1))), numpy.arange(1, xr+1), indexing='ij'))
	sy = numpy.concatenate((sy, asy))
	sx = numpy.concatenate((sx, asx))
	sr = numpy.sqrt((dy*sy)**2+(dx*sx)**2)
	sth = numpy.arctan2(sy, sx)
	
	rgains2 = numexpr.evaluate("1/sr**2*(2-2*cos(kr*sr*cos(th+sth)))/kr**2", local_dict=dict(kr=kr[None,:,:], th=th[None,:,:], sr=sr[:,None,None], sth=sth[:,None,None]))
	def rgain2(p):
		return numexpr.evaluate("sum(p*rgains, 0)", local_dict=dict(rgains=rgains2, p=p[:,None,None]))
	
	def tomin2(p):
		a = numpy.linalg.norm(rgain2(p)-ref)
		return a
	
	fit2 = scipy.optimize.minimize(tomin2, numpy.concatenate((pref, numpy.zeros(asy.size))), method='Powell')
	
	p = fit2.x
	p /= pref.sum()/2
	
	kernel = numpy.zeros(y.shape+x.shape)
	for i in range(p.size):
		kernel[(yr,yr-sy[i],yr+sy[i]),(xr,xr-sx[i],xr+sx[i])] += p[i]/sr[i]**2*numpy.array((-2,1,1))
		
	return kernel

__kernels = {}
def get_laplace_kernel(yr, dy=1, dx=None, xr=None):
	if xr is None: xr=yr
	if dx is None: dx=dy
	
	if (yr,xr,dy,dx) not in __kernels:
		__kernels.update({(yr,xr,dy,dx):generate_laplace_kernel(yr, dy=dy, dx=dx, xr=xr)})
	return __kernels[(yr,xr,dy,dx)]
