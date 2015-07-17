from __future__ import division, print_function, absolute_import 

import numpy
import numexpr
import scipy.special
import scipy.interpolate
import scipy.ndimage

from ...Utilities import Physics
from ...Mathematics import CoordinateTrafos as CT
from ...Utilities import MemArray as MA
from ...Utilities import Progress
from ...Utilities import HDFArray as HA
from ...Mathematics import FourierTransforms as FT
from ...Mathematics import Interpolator2D

from .Base import Kernel

def supersample(points, size, factor):
	return numpy.add.outer(numpy.require(points), numpy.linspace(-size/2*(1-1/factor),size/2*(1-1/factor), factor))

class FresnelKernel2(Kernel):
	ndims = 3
	dtype=numpy.float64
	itype=numpy.int32

	def __init__(self, z,y,x,t,e, k, focus=0, mask=None):
		self.__arrays__ = ['dat','idx_te','idx_yx','bounds','z','y','x','t','e','mask']

		self.__dict__.update(dict(z=z, y=y, x=x, t=t, e=e, k=k, focus=focus))

		self.idz = self.estimate_idz_range()
		self.dz = (self.z[1]-self.z[0])*self.idz

		if mask:
			self.mask = numpy.add.outer((numpy.arange(y.size)-y.size//2)**2,(numpy.arange(x.size)-x.size//2)**2).flatten()<(.25*min(y.size**2, x.size**2))
		else:
			self.mask = numpy.ones(y.size*x.size, numpy.bool)
			
	@property
	def shape(self):
		return self.idz.shape+self.t.shape+self.e.shape+self.idz.shape+self.y.shape+self.x.shape
	
	@property
	def fshape(self):
		return (self.idz.size*self.t.size*self.e.size, self.idz.size*self.y.size*self.x.size)

	def estimate_idz_range(self):
		m_w = max(numpy.amax(self.x)-self.focus,-numpy.amin(self.x)+self.focus,numpy.amax(self.y)-self.focus,-numpy.amin(self.y)+self.focus)
		m_dz = int(numpy.ceil(numpy.sqrt(numpy.pi*2*m_w/self.k)/(self.z[1]-self.z[0])))
		return numpy.arange(-m_dz,m_dz+1, dtype=self.itype)

	def calc(self):
		nnc_mask = numpy.count_nonzero(self.mask)
		
		e_idx = numpy.ndarray((self.e.size, nnc_mask), dtype=self.itype)
		e_idx[...] = numpy.mgrid[:self.e.size][:,None]
		e_idx = e_idx.flatten()
		
		yx_idx = numpy.ndarray((self.e.size, nnc_mask), dtype=self.itype)
		yx_idx[...] = numpy.mgrid[:self.y.size*self.x.size][self.mask][None,:]
		yx_idx = yx_idx.flatten()

		sel = numpy.empty(self.e.size*self.y.size*self.x.size, dtype=numpy.bool)

		wvs = []
		wmn = numpy.inf
		wmx = -numpy.inf

		for it, ti in Progress(enumerate(self.t), self.t.size, True):
			trafo = CT.Trafo2D(rad=ti)
			w,v,o = trafo.apply_to_bases(self.y,self.x)
			w,v = numexpr.evaluate("w+v+o", local_dict=dict(w=w[:,:,None],v=v[:,None,:],o=o[:,None,None])).reshape(2, self.y.size*self.x.size)
			w -= self.focus
			wmn = min(wmn, numpy.amin(w))
			wmx = max(wmn, numpy.amax(w))
			wvs.append((w,v))

		dpw = min(self.y[1]-self.y[0],self.x[1]-self.x[0])
		dpv = min(self.y[1]-self.y[0],self.x[1]-self.x[0])
		dpu = self.z[1]-self.z[0]
		
		ssv = 8
		ssu = 4
		ssw = 4
		
		wimn = numpy.floor(wmn/dpw)
		wimx = numpy.ceil(wmx/dpw)
		vimx = numpy.ceil(numpy.sqrt(2*numpy.pi*max(wmx,-wmn)/self.k)/dpv)
		uimx = numpy.ceil(numpy.sqrt(2*numpy.pi*max(wmx,-wmn)/self.k)/dpu)
		
		self.idz = numpy.arange(-ssu*uimx, ssu*uimx+1, dtype=self.itype)
		self.dz = dpu*self.idz
		
		pu = supersample(self.dz, dpu, ssu).flatten()
		pv = supersample(dpv*numpy.arange(-ssv*vimx, ssv*vimx+1, dtype=self.itype), dpv, ssv).flatten()
		pw = supersample(dpw*numpy.arange(wimn, wimx+1, dtype=self.itype), dpw, ssw).flatten()
		
		kpu, kpv = FT.reciprocal_coords(pu,pv)
		
		propf = numexpr.evaluate("-j*k/(pi*(ku**2+kv**2))*(exp(-j*(w-dw/2)/(2*k)*(ku**2+kv**2))-exp(-j*(w+dw/2)/(2*k)*(ku**2+kv**2)))", local_dict=dict(j=1j, k=self.k, pi=numpy.pi, w=pw[:,None,None], ku=kpu[None,:,None], kv=kpv[None,None,:], dw=dpw))
		
		propf[:,kpu==0,kpv==0] = dpw/(2*numpy.pi)

		upxmask = numpy.zeros(pu.shape, dtype=propf.dtype)
		upxmask[pu.size//2-(ssu//2):pu.size//2+(ssu+1)//2] = 1
		ufpxmask = FT.fft(upxmask)

		vpxmask = numpy.zeros(pv.shape, dtype=propf.dtype)
		vpxmask[pv.size//2-(ssv//2):pv.size//2+(ssv+1)//2] = 1
		vfpxmask = FT.fft(vpxmask)
		
		propf = numexpr.evaluate("propf*um*vm", local_dict=dict(propf=propf, um=ufpxmask[None,:,None], vm=vfpxmask[None,None,:]))

		prop = FT.ifft(propf, axes=(1,2)).real.astype(self.dtype)

		prop = scipy.ndimage.filters.convolve1d(prop, 1/ssw*numpy.ones(ssw, prop.dtype), axis=0)
		
		del propf
	
		prop = prop[:, pu.size//2-(ssu*uimx)-ssu//2:pu.size//2+(ssu*uimx)+(ssu//2), pv.size//2-(ssv*vimx)-ssv//2:pv.size//2+(ssv*vimx)+(ssv//2)]
		prop = prop.reshape(prop.shape[0], 2*uimx+1, ssu, prop.shape[2]).mean(2)
		pu = pu[pu.size//2-(ssu*uimx)-ssu//2:pu.size//2+(ssu*uimx)+(ssu//2)].reshape(2*uimx+1, ssu).mean(1)
		pv = pv[pv.size//2-(ssv*vimx)-ssv//2:pv.size//2+(ssv*vimx)+(ssv//2)]
		
		#sel = numexpr.evaluate('pi>=(v**2-ssv*dv**2+u**2-ssu*du**2)*k/(2*abs(w))', local_dict=dict(ssu=ssu, ssv=ssv, w=pw[:,None,None], u=pu[None,:,None], v=pv[None,None,:], pi=numpy.pi, k=self.k, du=pu[1]-pu[0], dv=pv[1]-pv[0]))
		
		#prop[~sel] = 0
		
		dat_z = HA.HDFConcatenator(self.dtype)
		te_z = HA.HDFConcatenator(self.itype)
		yx_z = HA.HDFConcatenator(self.itype)
		
		dyx = dpw

		self.idz = self.idz[self.idz.size//2-uimx:self.idz.size//2+uimx+1]
		self.dz = dpu*self.idz
		
		for iz, zi in Progress(enumerate(self.dz), self.idz.size, True):
			ipz = numpy.argmin(numpy.abs(pu-zi))
			prop_interpolator = Interpolator2D(pw, pv, prop[:,ipz,:])
			
			idat_z = [numpy.ndarray(0, self.dtype)]
			ite_z = [numpy.ndarray(0, self.itype)]
			iyx_z = [numpy.ndarray(0, self.itype)]

			for it, ti in Progress(enumerate(self.t), self.t.size):
				w,v = wvs[it]

				vmemax = numpy.sqrt(numpy.pi/self.k*2*max(wmx,-wmn)+dyx**2-zi**2)
				
				sel = numexpr.evaluate('vmemax>=abs(v-e)', local_dict=dict(v=v[yx_idx],  e=self.e[e_idx], vmemax=vmemax))
				sel[sel] = numexpr.evaluate('pi>=((v-e)**2-dv**2+z**2)*k/(2*abs(w))', local_dict=dict(z=zi, e=self.e[e_idx[sel]], w=w[yx_idx[sel]], v=v[yx_idx[sel]], pi=numpy.pi, k=self.k, dw=dyx, dv=dyx))
				
				#sel = numexpr.evaluate('pi>=((v-e)**2-dv**2+z**2)*k/(2*abs(w))', local_dict=dict(z=zi, e=self.e[e_idx], w=w[yx_idx], v=v[yx_idx], pi=numpy.pi, k=self.k, dw=dyx, dv=dyx))
				
				if numpy.count_nonzero(sel)>0:

					e_sel = e_idx[sel]
					yx_sel = yx_idx[sel]

					dat = prop_interpolator(w[yx_sel], v[yx_sel]-self.e[e_sel])

					psel = dat>0

					idat_z.append(numpy.require(dat[psel], requirements='O'))
					ite_z.append(numpy.require(it*self.e.size+e_sel[psel], requirements='O'))
					iyx_z.append(numpy.require(yx_sel[psel], requirements='O'))

                    
			dat_z.append(numpy.concatenate(idat_z))
			del idat_z
			te_z.append(numpy.concatenate(ite_z))
			del ite_z
			yx_z.append(numpy.concatenate(iyx_z))
			del iyx_z

		del dat, sel, trafo, e_idx, yx_idx, w,v,o

		self.bounds = [0]+dat_z.sizes
		self.bounds = numpy.cumsum(self.bounds).astype(self.itype)

		self.dat = dat_z.concatenate()#numpy.concatenate(dat_z)
		self.idx_te = te_z.concatenate()#numpy.concatenate(te_z)
		self.idx_yx = yx_z.concatenate()#numpy.concatenate(yx_z)

		return (self.dat, self.idx_te, self.idx_yx, self.bounds)

	@property
	def idx(self):
		return (self.idx_te, self.idx_yx)

