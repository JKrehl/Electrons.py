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
from ...Utilities import Magic
from ...Utilities import HDFArray as HA
from ...Mathematics import FourierTransforms as FT
from ...Mathematics import Interpolator2D

from .Base import Kernel

def supersample(points, size, factor):
	return numpy.add.outer(numpy.require(points), numpy.linspace(-size/2,size/2, factor, False))

def create_convolution_kernel(trafo, h, w):
	size = 2*numpy.ceil(.71*max(h, w))+1
	window = numpy.ones((h,w))/(h*w)

	trafo = trafo.copy()
	trafo.postshift([size/2, size/2])
	trafo.preshift([-h/2, -w/2])

	window = scipy.ndimage.interpolation.affine_transform(window, trafo.affine, offset=trafo.offset, output_shape=(size, size), order=1)

	while numpy.all(window[(0,-1),:]==0): window = window[1:-1, :]
	while numpy.all(window[:,(0,-1)]==0): window = window[:, 1:-1]

	return window

class FresnelKernel3(Kernel):
	ndims = 3
	dtype=numpy.complex64
	itype=numpy.int32

	def __init__(self, z,y,x,t,d, k, focus=0, bandlimit=0,  mask=None):
		self.__arrays__ = ['dat','idx_te','idx_yx','bounds','z','y','x','t','d','focus','mask']

		self.__dict__.update(dict(z=z, y=y, x=x, t=t, d=d, k=k, bandlimit=bandlimit, focus=focus))

		if not isinstance(self.focus, numpy.ndarray):
			self.focus = self.focus*numpy.ones(self.t.shape, numpy.obj2sctype(self.focus))
		
		if mask:
			self.mask = numpy.add.outer((numpy.arange(y.size)-y.size//2)**2,(numpy.arange(x.size)-x.size//2)**2).flatten()<(.25*min(y.size**2, x.size**2))
		else:
			self.mask = numpy.ones(y.size*x.size, numpy.bool)
			
	@property
	def shape(self):
		return self.idz.shape+self.t.shape+self.d.shape+self.idz.shape+self.y.shape+self.x.shape
	
	@property
	def fshape(self):
		return (self.idz.size*self.t.size*self.d.size, self.idz.size*self.y.size*self.x.size)

	def calc(self):
		dw = self.z[1]-self.z[0]
		dv = min(self.y[1]-self.y[0],self.x[1]-self.x[0])
		du = dv

		if self.bandlimit==0:
			self.bandlimit = dw
		
		ss_v = 4
		ss_u = 4
		cutoff = 1e-2
		
		vus = []
		kernels = []
		v_min = numpy.inf
		v_max = -numpy.inf

		for it, ti in Progress(enumerate(self.t), self.t.size, True):
			trafo = CT.Trafo2D().rad(ti)
			v,u,o = trafo.apply_to_bases(self.y,self.x)
			v,u = numexpr.evaluate("v+u+o", local_dict=dict(v=v[:,:,None],u=u[:,None,:],o=o[:,None,None])).reshape(2, self.y.size*self.x.size)
			v -= self.focus[it]
			v_min = min(v_min, numpy.amin(v))
			v_max = max(v_max, numpy.amax(v))
			vus.append((v,u))

			kern = create_convolution_kernel(trafo, ss_v, ss_u)
			kern = kern.reshape((1,)+kern.shape)
			kernels.append(kern)

		v_amax = max(-v_min, v_max)
		rho_max = -v_amax*self.bandlimit/dv+numpy.sqrt((v_amax*self.bandlimit/dv)**2+6*numpy.pi*2*v_amax**2/(self.k*dv))
		
		w_i = numpy.arange(-numpy.ceil(rho_max/dw), numpy.ceil(rho_max/dw))
		w = dw*w_i
		v = dv*numpy.arange(numpy.floor(v_min/dv), numpy.ceil(v_max/dv))
		u = du*numpy.arange(-numpy.ceil(rho_max/du), numpy.ceil(rho_max/du))
		
		v_ss = supersample(v, dv, ss_v).flatten()
		u_ss = supersample(u, du, ss_u).flatten()
		
		kw = FT.reciprocal_coords(w)
		ku_ss = FT.reciprocal_coords(u_ss)

		propf = numexpr.evaluate("1/(2*pi)*exp(-1j*v/(2*k)*(kw**2+ku**2))", local_dict=dict(j=1j, pi=numpy.pi, k=self.k, kw=kw[:,None,None], v=v_ss[None,:,None], ku=ku_ss[None,None,:]))
		propf = FT.mwedge(propf, axes=(0,2))

		kr = numexpr.evaluate("sqrt(kw**2+ku**2)*dr", local_dict=dict(kw=kw[:,None], ku=ku_ss[None,:], dr=self.bandlimit, pi=numpy.pi))
		res_win = numexpr.evaluate("where(kr==0, 1, 2*j1kr/kr)", local_dict=dict(kr=kr, j1kr=scipy.special.j1(kr)))
		
		propf = numexpr.evaluate("propf*res_win", local_dict=dict(propf=propf, res_win=res_win[:,None,:]))
		prop = FT.ifft(propf, axes=(0,2))

		del propf
		
		prop_max = numpy.amax(numpy.abs(prop))
		prop_ez = prop < cutoff*prop_max

		prop_shrink = numpy.array([[Magic.where_first_not(a), Magic.where_last_not(a)] for a in (numpy.all(prop_ez, axis=(1,2)), numpy.all(prop_ez, axis=(0,1)))])
		prop_shrink[0] += [-1,+1]
		prop_shrink[1] += [-ss_u, +ss_u]
		prop_shrink[prop_shrink<0] = 0
		
		prop = numpy.require(prop[prop_shrink[0,0]:prop_shrink[0,1], :, prop_shrink[1,0]:prop_shrink[1,1]], None, 'O')
		w_i_sh = w_i[prop_shrink[0,0]:prop_shrink[0,1]]
		w_sh = w[prop_shrink[0,0]:prop_shrink[0,1]]
		u_ss_sh = u_ss[prop_shrink[1,0]:prop_shrink[1,1]]

		return prop
		
		del kr, res_win
		
		nnc_mask = numpy.count_nonzero(self.mask)
		
		d_idx = numpy.ndarray((self.d.size, nnc_mask), dtype=self.itype)
		d_idx[...] = numpy.mgrid[:self.d.size][:,None]
		d_idx = d_idx.flatten()
		
		yx_idx = numpy.ndarray((self.d.size, nnc_mask), dtype=self.itype)
		yx_idx[...] = numpy.mgrid[:self.y.size*self.x.size][self.mask][None,:]
		yx_idx = yx_idx.flatten()
		
		dat_z = HA.HDFConcatenator(self.dtype)
		td_z = HA.HDFConcatenator(self.itype)
		yx_z = HA.HDFConcatenator(self.itype)
		
		for iz, zi in Progress(enumerate(w_i_sh), w_i_sh.size, True):
			

			idat_z = [numpy.ndarray(0, self.dtype)]
			itd_z = [numpy.ndarray(0, self.itype)]
			iyx_z = [numpy.ndarray(0, self.itype)]
			
			for it, ti in Progress(enumerate(self.t), self.t.size):
				v,u = vus[it]
				kern = kernels[it]
				
				propi = scipy.ndimage.filters.convolve(propf.real, kern, mode='constant') + 1j*scipy.ndimage.filters.convolve(propf.imag, kern, mode='constant')
				shrinki = [Magic.where_first(numpy.all(propi_nz[iz, :, :], axis=1)), Magic.where_last(numpy.all(propi_nz[iz, :, :], axis=1))+1]

				interpi = lambda x: scipy.interpolate.interpn((v_ss, u_ss_sh[shrinki[0]:shrinki[1]]), propi[iz, :, shrinki[0]:shrinki[1]], x, method='linear', bounds_error=False, fill_value=0)

				dif_max = max(-u_ss_sh[shrinki[0]], u_ss_sh[shrinki[1]])
				sel = numexpr.evaluate("abs(u-d)<=dif_max", local_dict=dict(u=u[yx_idx], d=d[d_idx], dif_max=dif_max))
				
				if numpy.count_nonzero(sel)>0:
					d_sel = d_idx[sel]
					yx_sel = yx_idx[sel]

					dat = interpi(numpy.hstack((v[yx_sel], numexpr.evaluate("u-d", local_dict=dict(u=u[yx_sel], d=self.d[d_sel])))))

					psel = dat >= cutoff*prop_max

					idat_z.append(numpy.require(dat[psel], requirements='O'))
					itd_z.append(numpy.require(it*self.d.size+d_sel[psel], requirements='O'))
					iyx_z.append(numpy.require(yx_sel[psel], requirements='O'))

				dat_z.append(numpy.concatenate(idat_z))
				del idat_z
				td_z.append(numpy.concatenate(itd_z))
				del itd_z
				yx_z.append(numpy.concatenate(iyx_z))
				del iyx_z

			del propi, sel, dat, psel, d_sel, yx_sel, d_idx, yx_idx, v_u

			self.bounds = [0]+dat_z.sizes
			self.bounds = numpy.cumsum(self.bounds).astype(self.itype)

			self.dat = dat_z.concatenate()
			self.idx_td = td_z.concatenate()
			self.idx_yx = yx_z.concatenate()
			
			return (self.dat, self.idx_td, self.idx_yx, self.bounds)
		
	@property
	def idx(self):
		return (self.idx_td, self.idx_yx)

