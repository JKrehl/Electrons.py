from __future__ import division, print_function

import numpy
import numexpr

from .Base import Kernel

class RayKernel(Kernel):
	def __init__(self, y, x, t, d, mask=None, dtype=None, lazy=False):
		y,x,t,d = (numpy.require(i) for i in (y,x,t,d))
		self.__dict__.update(dict(y=y, x=x, t=t, d=d, dtype=dtype))
		self.ndims = 2
		self.shape = t.shape+d.shape+y.shape+x.shape
		self.fshape = (t.size*d.size, y.size*x.size)

		if mask:
			self.mask = numpy.add.outer((numpy.arange(y.size)-y.size//2)**2,(numpy.arange(x.size)-x.size//2)**2).flatten()<(.25*min(y.size**2, x.size**2))
		else:
			self.mask = None
		
		if not lazy:
			self.calc()
		
	def calc(self):
		res = [[],[],[]]

		xd = abs(self.x[1]-self.x[0])
		yd = abs(self.y[1]-self.y[0])
		dd = abs(self.d[1]-self.d[0])
		
		unit_area = xd*yd/dd

		x = self.x/xd
		y = self.y/yd
		d = self.d/dd
		
		idx = numpy.indices((self.d.size, self.y.size*self.x.size))
		
		for it,ti in enumerate(self.t):
			al = abs((ti+numpy.pi/4)%(numpy.pi/2) - numpy.pi/4)
			a = .5*(numpy.cos(al)-numpy.sin(al))
			b = .5*(numpy.cos(al)+numpy.sin(al))
			h = 1/numpy.cos(al)
			
			if b==a: f = 0
			else: f = h/(b-a)
			
			e = numexpr.evaluate("x*cos(t)-y*sin(t) -d", local_dict=dict(x=x[None,None,:], y=y[None,:,None], d=d[:,None,None], t=ti)).reshape(d.size, y.size*x.size)
			if self.mask is not None:
				sel = numexpr.evaluate("mask&(abs(e)<(b+.5))", local_dict=dict(mask = self.mask[None,:], e=e, b=b))
			else:
				sel = numexpr.evaluate("(abs(e)<(b+.5)", local_dict=dict(e=e, b=b))

			e = e[sel]
			sel = idx[:,sel]
			
			calcs = 'where({0}<-b, 0, where({0}<-a, .5*f*({0}+b)**2, where({0}<a, .5*f*(a-b)**2+h*({0}+a), where({0}<b, 1-.5*f*(b-{0})**2, 1))))'
			ker = numexpr.evaluate('area*('+calcs.format('(dif+.5)')+'-'+calcs.format('(dif-.5)')+')', local_dict=dict(dif=e, a=a, b=b, f=f, h=h, area=unit_area))
			
			csel = ker>=0#1e-2
			sel = sel[:,csel]
			
			res[0].append(ker[csel])
			res[1].append(sel[0,:]+self.d.size*it)
			res[2].append(sel[1,:])

			assert res[0][-1].size == res[1][-1].size

		self.dat = numpy.hstack(res[0])
		self.idx = (numpy.hstack(res[2]), numpy.hstack(res[1]))
			
		self.status = 0

		return (self.dat, self.idx[0], self.idx[1])
