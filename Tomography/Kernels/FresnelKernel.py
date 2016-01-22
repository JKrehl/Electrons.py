 
import numpy
import numexpr
import scipy.special
import scipy.interpolate
import scipy.ndimage

from ...Utilities import Physics
from ...Mathematics import CoordinateTrafos as CT
from ...Utilities import Progress
from ...Utilities import Magic
from ...Mathematics import FourierTransforms as FT
from ...Mathematics import Interpolator2D

from .Kernel import Kernel

import gc

def supersample(points, size, factor):
	return numpy.add.outer(numpy.require(points), numpy.linspace(0, size, factor, False))

class FresnelKernel(Kernel):
	def __init__(self, path=None, memory_strategy=1):
		if not hasattr(self, '_arrays'):
			self._arrays = {}

		self._arrays.update(z=0, y=0, x=0, t=0, d=0, focus=0, k=2, mask=2, dtype=2, itype=2, ss_w=2, ss_v=2, ss_u=2, cutoff=2, idz=0, bounds=0, dat=1, row=1, col=1)

		if memory_strategy == 0:
			self._arrays = {key:0 if val==1 else val for key,val in self._arrays.items()}
		elif memory_strategy == 1:
			pass
		elif memory_strategy == 2:
			self._arrays = {key:1 if val==0 else val for key,val in self._arrays.items()}
		else:
			raise ValueError

		super().__init__(path, memory_strategy)

	def init(self, z,y,x,t,d,
	         k, focus=0, bandlimit=0,mask=None,
			 dtype=numpy.complex64, itype=numpy.int32,
			 ss_w=2, ss_v=1, ss_u=2, cutoff=1e-2,
			 ):

		self.init_empty_arrays()

		if not isinstance(focus, numpy.ndarray) or focus.size==1:
			focus = focus*numpy.ones(t.shape, numpy.obj2sctype(focus))

		self.z, self.y, self.x = z, y, x
		self.t, self.d = t, d
		self.k, self.bandlimit, self.focus, self.mask = k, bandlimit, focus, mask
		self.dtype, self.itype = dtype, itype
		self.ss_w, self.ss_v, self.ss_u, self.cutoff = ss_w, ss_v, ss_u, cutoff

		return self

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

		rho_max = numexpr.evaluate("sqrt(-2*bl**2*z**2/dz**2+2*z**2/(k*dz**2)*sqrt(bl**4*k**2+dp**2*dz**2))", local_dict=dict(bl=self.bandlimit, k=self.k, dz=dv, z=max(-v_min, v_max), dp=4*numpy.pi))

		w_i = numpy.arange(-int(numpy.ceil(rho_max/dw)), int(numpy.ceil(rho_max/dw)), dtype=self.itype)
		w = dw*w_i
		v = dv*numpy.arange(int(numpy.floor(v_min/dv)), int(numpy.ceil(v_max/dv)))
		u = du*numpy.arange(-int(numpy.ceil(rho_max/du)), int(numpy.ceil(rho_max/du)))
		
		w_ss = supersample(w, dw, self.ss_w).flatten()
		v_ss = supersample(v, dv, self.ss_v).flatten()
		u_ss = supersample(u, du, self.ss_u).flatten()

		kw_ss = FT.mreciprocal_coords(w_ss)
		ku_ss = FT.mreciprocal_coords(u_ss)

		propf = numexpr.evaluate('1/(2*pi)*exp(1j*(-v/(2*k)*(kw**2+ku**2)))', local_dict=dict(j=1j, pi=numpy.pi, k=self.k, kw=kw_ss[:, None, None], v=v_ss[None, :, None], ku=ku_ss[None, None, :]))

		#kr = numexpr.evaluate("sqrt(kw**2+ku**2)*dr", local_dict=dict(kw=kw_ss[:,None], ku=ku_ss[None,:], dr=self.bandlimit, pi=numpy.pi))
		#res_win = numexpr.evaluate('where(kr==0, 1, 2*j1kr/kr)', local_dict=dict(kr=kr, j1kr=scipy.special.j1(kr)))
		#res_win /= numpy.mean(res_win)
		
		#propf = numexpr.evaluate("propf*res_win", local_dict=dict(propf=propf, res_win=res_win[:,None,:]))

		prop = numpy.empty(w.shape + v_ss.shape + u_ss.shape, self.dtype)
		for i in range(prop.shape[1]):
			prop[:, i, :] = numpy.roll(FT.mifft(propf[:, i, :]), (self.ss_w//2), 0).reshape(w.size, self.ss_w, u_ss.size).mean(1)

		sel_rho_max = numexpr.evaluate("sqrt(-2*bl**2*z**2/dz**2+2*z**2/(k*dz**2)*sqrt(bl**4*k**2+dp**2*dz**2))", local_dict=dict(bl=self.bandlimit, k=self.k, dz=dv, z=v_ss, dp=numpy.pi))
		prop = numexpr.evaluate('where(ww+uu<=rr_max+dww+duu, prop, 0)', local_dict=dict(prop=prop, ww=w[:, None, None]**2, dww=dw**2/4, uu=u_ss[None,None,:]**2, duu=du**2/4,rr_max=sel_rho_max[None, :, None]**2))

		del propf#, kr, rr, res_win
		
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

		if self.mask:
			mask = numpy.add.outer((numpy.arange(self.y.size)-self.y.size//2)**2,(numpy.arange(self.x.size)-self.x.size//2)**2).flatten()<(.25*min(self.y.size**2, self.x.size**2))
		else:
			mask = numpy.ones(self.y.size*self.x.size, numpy.bool)

		nnc_mask = numpy.count_nonzero(mask)

		d_idx = numpy.repeat(numpy.mgrid[:self.d.size], nnc_mask)
		yx_idx = numpy.repeat(numpy.mgrid[:self.y.size*self.x.size][mask].reshape(1, nnc_mask), self.d.size, 0).flatten()

		del mask

		dat_concatenator = self.arrays.dat.concatenator(self.dtype)
		row_concatenator = self.arrays.row.concatenator(self.itype)
		col_concatenator = self.arrays.col.concatenator(self.itype)

		bounds = []

		for iz, zi in Progress(enumerate(w_i_sh), w_i_sh.size, True):
			entries = 0
			
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

						dat_concatenator.append(dat[psel])
						row_concatenator.append(d_sel[psel] + it*self.d.size)
						col_concatenator.append(yx_sel[psel])

						entries += numpy.count_nonzero(psel)

						del d_sel, yx_sel, dat, psel

					del shrinki, interpi, diff, sel

				del propi, propi_nz
				del v,u

			bounds.append(entries)
			
		del prop, yx_idx, d_idx

		gc.collect()
		
		self.bounds = numpy.array(bounds)
		sel = self.bounds > 0
		self.bounds = numpy.cumsum(numpy.hstack((0, self.bounds[sel]))).astype(self.itype)

		self.idz = w_i_sh[sel]

		dat_concatenator.finalize()
		del dat_concatenator
		row_concatenator.finalize()
		del row_concatenator
		col_concatenator.finalize()
		del col_concatenator
		
		return self