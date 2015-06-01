from __future__ import division, print_function, absolute_import 

import numpy
import numexpr

from Physics.Utilities import Physics, CoordinateTrafos as CT, MemArray as MA, Progress

from .Base import Kernel

def supersample(points, size, factor):
	return numpy.add.outer(points, numpy.linspace(-size/2*(1-1/factor),size/2*(1-1/factor), factor))

class FresnelKernel(Kernel):
	def __init__(self, z,y,x,t,e,k,focus=0, dat_t=numpy.float64, idx_t=numpy.int32, mask=None):
		self.__dict__.update(dict(z=z,y=y,x=x,t=t,e=e,k=k,focus=focus,dat_t=dat_t,idx_t=idx_t))
		self.ndims = 3

		self.idz = self.estimate_idz_range()
		self.dz = (self.z[1]-self.z[0])*self.idz
		
		self.shape = self.idz.shape+self.t.shape+self.e.shape+self.idz.shape+self.y.shape+self.x.shape
		self.fshape = (self.idz.size*self.t.size*self.e.size, self.idz.size*self.y.size*self.x.size)

		if mask:
			self.mask = numpy.add.outer((numpy.arange(y.size)-y.size//2)**2,(numpy.arange(x.size)-x.size//2)**2).flatten()<(.25*min(y.size**2, x.size**2))
		else:
			self.mask = numpy.ones(y.size*x.size, numpy.bool)
		

	def estimate_idz_range(self):
		m_w = max(numpy.amax(self.x)-self.focus,-numpy.amin(self.x)+self.focus,numpy.amax(self.y)-self.focus,-numpy.amin(self.y)+self.focus)
		m_dz = int(numpy.ceil(numpy.sqrt(numpy.pi*2*m_w/self.k)/(self.z[1]-self.z[0])))
		return numpy.arange(-m_dz,m_dz+1, dtype=self.idx_t)
        
	def calc(self):
		e_idx, yx_idx = numpy.mgrid[:self.e.size,:self.y.size*self.x.size][:,:,self.mask].astype(self.idx_t)
		e_idx = e_idx.flatten()
		yx_idx = yx_idx.flatten()
	
		#dat = numpy.zeros((self.e.size, numpy.count_nonzero(self.mask)), dtype=self.dat_t)
		sel = numpy.empty(self.e.size*numpy.count_nonzero(self.mask), dtype=numpy.bool)
	
		ewvs = []
	
		for it, ti in Progress(enumerate(self.t), self.t.size):
			trafo = CT.Trafo2D(rad=ti)
			w,v,o = trafo.apply_to_bases(self.y,self.x)
			w,v = numexpr.evaluate("w+v+o", local_dict=dict(w=w[:,:,None],v=v[:,None,:],o=o[:,None,None])).reshape(2, self.y.size*self.x.size)[:,self.mask]
			w -= self.focus
			e = self.e.repeat(w.size)
			w = w[None,:].repeat(self.e.size, 0).flatten()
			v = v[None,:].repeat(self.e.size, 0).flatten()
			ewvs.append((e,w,v))
			
		dat_z = []
		te_z = []
		yx_z = []

		dyx = min(self.y[1]-self.y[0],self.x[1]-self.x[0])
		dz = self.z[1]-self.z[0]
		dA = (self.y[1]-self.y[0])*(self.x[1]-self.x[0])
		
		for iz, zi in Progress(enumerate(self.dz), self.idz.size, True):
			idat_z = []
			ite_z = []
			iyx_z = []
        
			for it, ti in Progress(enumerate(self.t), self.t.size):
				e,w,v = ewvs[it]

				numexpr.evaluate('pi>=abs(((v-e)**2+z**2)*k/(2*abs(w)))', local_dict=dict(z=zi, e=e, w=w, v=v, pi=numpy.pi, k=self.k, dw=dyx), out=sel)

				e_sel = e_idx[sel]
				yx_sel = yx_idx[sel]
				
				#dat = numexpr.evaluate("k/(2*pi*abs(w))*exp(j*k/(2*abs(w))*((v-e)**2+z**2))*dA",local_dict=dict(z=zi, v=v[yx_sel], w=w[yx_sel], e=self.e[e_sel], pi=numpy.pi, j=1j, k=self.k, dA=dA)).real
				dat = numexpr.evaluate("k/(2*pi*abs(w))*exp(j*k/(2*abs(w))*((v-e)**2+z**2))*dA", local_dict=dict(z=supersample(zi,dz,4)[None,:,None,None], v=supersample(v[yx_sel],dyx,4)[:,None,:,None], w=supersample(w[yx_sel],dyx,4)[:,None,None,:], e=e[e_sel][:,None,None,None], pi=numpy.pi, j=1j, k=self.k, dA=dA)).mean(axis=(1,2,3)).real
				
				idat_z.append(numpy.require(dat, requirements='O'))
				ite_z.append(numpy.require(it*self.e.size+e_sel, requirements='O'))
				iyx_z.append(numpy.require(yx_sel, requirements='O'))
				
				
			dat_z.append(MA.memconcatenate(idat_z))
			del idat_z
			te_z.append(MA.memconcatenate(ite_z))
			del ite_z
			yx_z.append(MA.memconcatenate(iyx_z))
			del iyx_z
    
		del dat, sel, trafo, e_idx, yx_idx, w,v,o
    
		self.bounds = [0]+[i.size for i in dat_z]
		self.bounds = numpy.cumsum(self.bounds).astype(self.idx_t)
    
		self.dat = numpy.concatenate(dat_z)
		self.idx_te = numpy.concatenate(te_z)
		self.idx_yx = numpy.concatenate(yx_z)

		self.idx = (self.idx_te, self.idx_yx)
		
		return [self.dat, self.idz, self.bounds, self.idx_te, self.idx_yx]
