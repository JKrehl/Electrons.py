from __future__ import division, print_function

import numpy
import numexpr

from .Base import Kernel

class RayKernel(Kernel):
	def __init__(self, y, x, t, d, dtype=None, lazy=False):
		self.__dict__.update(dict(y=y, x=x, t=t, d=d, dtype=dtype))
		self.ndims = 2
		self.shape = t.shape+d.shape+y.shape+x.shape
		self.fshape = (t.size*d.size, y.size*x.size)

		if not lazy:
			self.calc()
		
	def calc(self):
		res = [[],[],[]]

		dd = abs(self.d[1]-self.d[0])

		x = self.x/dd
		y = self.y/dd
		d = self.d/dd
		
		idx = numpy.indices((self.d.size, self.y.size*self.x.size))
		
		for it,ti in enumerate(self.t):
			al = abs((ti+numpy.pi/4)%(numpy.pi/2) - numpy.pi/4)
			a = .5*(numpy.cos(al)-numpy.sin(al))
			b = .5*(numpy.cos(al)+numpy.sin(al))
			h = 1/numpy.cos(al)
			
			e = numexpr.evaluate("x*cos(t)-y*sin(t)", local_dict=dict(x=x[None,:], y=y[:,None], t=ti)).flatten()
			sel = idx[:,numexpr.evaluate("abs(e-d)<1.5",local_dict=dict(e=e[None,:], d=d[:,None]))]
			
			e = e[sel[1,:]] - d[sel[0,:]]
			
			calcs = 'where({0}<-b, 0, where({0}<-a, .5*e*({0}+b)**2, where({0}<a, .5*e*(a-b)**2+h*({0}+a), where({0}<b, 1-.5*e*(b-{0})**2, 1))))'
			ker = numexpr.evaluate(calcs.format('(dif+.5)')+'-'+calcs.format('(dif-.5)'), local_dict=dict(dif=e, a=a, b=b, e=e, h=h))
			
			csel = ker>=1e-2
			sel = sel[:,csel]
			
			res[0].append(ker[csel])
			res[1].append(sel[0,:]+self.d.size*it)
			res[2].append(sel[1,:])

			assert res[0][-1].size == res[1][-1].size

		self.dat = numpy.hstack(res[0])
		self.idx = (numpy.hstack(res[2]), numpy.hstack(res[1]))
			
		self.status = 0

		return (self.dat, self.idx[0], self.idx[1])
