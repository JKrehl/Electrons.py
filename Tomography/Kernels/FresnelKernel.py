 
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

import gc

def supersample(points, size, factor):
	return numpy.add.outer(numpy.require(points), numpy.linspace(-size/2, size/2, factor, False))

class FresnelKernel(Kernel):
	ndims = 3

	def __init__(self, z,y,x,t,d,
				 k, focus=0, bandlimit=0,mask=None,
				 dtype=numpy.complex64, itype=numpy.int32,
				 ss_w=3, ss_v=1, ss_u=4, cutoff=1e-2,
				 ):
		
		self.__arrays__ = ['dat','idx_td','idx_yx','idz','bounds','z','y','x','t','d','focus','mask']

		self.__dict__.update(dict(z=z, y=y, x=x, t=t, d=d,
								  k=k, bandlimit=bandlimit, focus=focus,
								  dtype=dtype, itype=itype,
								  ss_w=ss_w, ss_v=ss_v, ss_u=ss_u, cutoff=cutoff,
								  ))

		if not isinstance(self.focus, numpy.ndarray) or self.focus.size==1:
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
			self.bandlimit = max(dw, dv)
		
		vus = []
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

		v_amax = max(-v_min, v_max)
		rho_max = -v_amax*self.bandlimit/dv+numpy.sqrt((v_amax*self.bandlimit/dv)**2+6*numpy.pi*2*v_amax**2/(self.k*dv))
		
		w_i = numpy.arange(-numpy.ceil(rho_max/dw), numpy.ceil(rho_max/dw)+1, dtype=self.itype)
		w = dw*w_i
		v = dv*numpy.arange(numpy.floor(v_min/dv), numpy.ceil(v_max/dv))
		u = du*numpy.arange(-numpy.ceil(rho_max/du), numpy.ceil(rho_max/du))
		
		w_ss = supersample(w, dw, self.ss_w).flatten()
		v_ss = supersample(v, dv, self.ss_v).flatten()
		u_ss = supersample(u, du, self.ss_u).flatten()
		
		kw_ss = FT.mreciprocal_coords(w_ss)
		ku_ss = FT.mreciprocal_coords(u_ss)

		propf = numexpr.evaluate('1/(2*pi)*exp(1j*(-v/(2*k)*(kw**2+ku**2)))', local_dict=dict(j=1j, pi=numpy.pi, k=self.k, kw=kw_ss[:, None, None], v=v_ss[None, :, None], ku=ku_ss[None, None, :]))

		kr = numexpr.evaluate("sqrt(kw**2+ku**2)*dr", local_dict=dict(kw=kw_ss[:,None], ku=ku_ss[None,:], dr=self.bandlimit, pi=numpy.pi))
		res_win = numexpr.evaluate('where(kr==0, 1, 2*j1kr/kr)', local_dict=dict(kr=kr, j1kr=scipy.special.j1(kr)))
		res_win /= numpy.mean(res_win)
		
		propf = numexpr.evaluate("propf*res_win", local_dict=dict(propf=propf, res_win=res_win[:,None,:]))

		prop = numpy.empty(w.shape + v_ss.shape + u_ss.shape, self.dtype)
		for i in range(prop.shape[1]):
			prop[:, i, :] = FT.mifft(propf[:, i, :]).reshape(w.size, self.ss_w, u_ss.size).mean(1)
			
		sel_rho_max = -numpy.abs(v_ss)*self.bandlimit/dv + numpy.sqrt((numpy.abs(v_ss)*self.bandlimit/dv)**2 + 3*numpy.pi*2*numpy.abs(v_ss)**2/(self.k*dv))
		rr = numexpr.evaluate('w**2-dw**2+u**2-du**2', local_dict=dict(w=w[:, None], u=u_ss[None, :], dw=dw, du=du))
		prop = numexpr.evaluate('where(rr<=r_max, prop, 0)', local_dict=dict(prop=prop, rr=rr[:, None, :], r_max=sel_rho_max[None, :, None]**2))
		
		del propf, kr, rr, res_win
		
		prop_max = numpy.amax(numpy.abs(prop))
		prop_nz = numpy.abs(prop) >= self.cutoff*prop_max

		prop_shrink = numpy.array([[Magic.where_first(a), Magic.where_last(a)+1] for a in (numpy.any(prop_nz, axis=(1,2)), numpy.any(prop_nz, axis=(0,1)))])
		prop_shrink[0] += [-1,+1]
		prop_shrink[1] += [-self.ss_u, +self.ss_u]
		
		if prop_shrink[0,0] < 0: prop_shrink[0,0] = 0
		if prop_shrink[1,0] < 0: prop_shrink[1,0] = 0
		if prop_shrink[0,1] > prop_nz.shape[0]: prop_shrink[0,1] = prop_nz.shape[0]
		if prop_shrink[1,1] > prop_nz.shape[2]: prop_shrink[1,1] = prop_nz.shape[2]
		
		prop = numpy.require(prop[prop_shrink[0,0]:prop_shrink[0,1], :, prop_shrink[1,0]:prop_shrink[1,1]], None, 'O')
		w_i_sh = w_i[prop_shrink[0,0]:prop_shrink[0,1]]
		w_sh = w[prop_shrink[0,0]:prop_shrink[0,1]]
		u_ss_sh = u_ss[prop_shrink[1,0]:prop_shrink[1,1]]
		
		nnc_mask = numpy.count_nonzero(self.mask)

		d_idx = numpy.repeat(numpy.mgrid[:self.d.size], nnc_mask)
		yx_idx = numpy.repeat(numpy.mgrid[:self.y.size*self.x.size][self.mask].reshape(1, nnc_mask), self.d.size, 0).flatten()
		
		dat_z = HA.HDFConcatenator(self.dtype)
		td_z = HA.HDFConcatenator(self.itype)
		yx_z = HA.HDFConcatenator(self.itype)
		
		for iz, zi in Progress(enumerate(w_i_sh), w_i_sh.size, True):
			
			idat_z = [numpy.ndarray(0, self.dtype)]
			itd_z = [numpy.ndarray(0, self.itype)]
			iyx_z = [numpy.ndarray(0, self.itype)]
			
			for it, ti in Progress(enumerate(self.t), self.t.size):
				v,u = vus[it]
				
				propi = prop[iz, :, :]
				propi_nz = numpy.abs(propi) >= self.cutoff*prop_max

				if numpy.any(propi_nz):
					shrinki = [Magic.where_first(numpy.any(propi_nz, axis=0)), Magic.where_last(numpy.any(propi_nz, axis=0))+1]
					interpi = scipy.interpolate.RegularGridInterpolator((v_ss, u_ss_sh[shrinki[0]:shrinki[1]]), propi[:, shrinki[0]:shrinki[1]], method='linear', bounds_error=False, fill_value=None)
					
					diff_max = max(-u_ss_sh[shrinki[0]], -u_ss_sh[shrinki[1]-1])
					diff = numexpr.evaluate("u-d", local_dict=dict(u=u[yx_idx], d=self.d[d_idx]))
					sel = numexpr.evaluate("abs(diff)<=diff_max", local_dict=dict(diff=diff, diff_max=diff_max))
				
					if numpy.any(sel):
						d_sel = d_idx[sel]
						yx_sel = yx_idx[sel]
						diff = diff[sel]
						
						dat = interpi(numpy.vstack((v[yx_sel], diff)).T)
						
						psel = numpy.abs(dat) >= self.cutoff*prop_max
						
						idat_z.append(numpy.require(dat[psel], requirements='O'))
						itd_z.append(numpy.require(it*self.d.size+d_sel[psel], requirements='O'))
						iyx_z.append(numpy.require(yx_sel[psel], requirements='O'))

						del d_sel, yx_sel, dat, psel

					del shrinki, interpi, diff, sel

				del propi, propi_nz
				del v,u

			dat_z.append(numpy.concatenate(idat_z))
			td_z.append(numpy.concatenate(itd_z))
			yx_z.append(numpy.concatenate(iyx_z))
			del idat_z, itd_z, iyx_z
			
		del prop, yx_idx, d_idx

		gc.collect()
		
		self.bounds = numpy.array(dat_z.sizes)
		sel = self.bounds > 0
		self.bounds = numpy.cumsum(numpy.hstack((0, self.bounds[sel]))).astype(self.itype)

		self.idz = w_i_sh[sel]
		
		print('data array size: %f %s' % Magic.humanize_filesize(dat_z.size))
		print('td indices array size: %f %s' % Magic.humanize_filesize(td_z.size))
		print('yx indices array size: %f %s' % Magic.humanize_filesize(yx_z.size))

		self.dat = dat_z.get_array()
		del dat_z
		
		self.idx_td = td_z.get_array()
		del td_z
		
		self.idx_yx = yx_z.get_array()
		del yx_z
		
		return self
	
	@property
	def idx(self):
		return (self.idx_td, self.idx_yx)
