from __future__ import division, print_function, absolute_import 

import numpy
import numexpr
import scipy.special

import h5py
import os.path, cPickle, math

from Physics.Utilities import Physics, CoordinateTrafos as CT, MemArray as MA, Progress

from .Base import Kernel

def supersample(points, size, factor):
	return numpy.add.outer(numpy.require(points), numpy.linspace(-size/2*(1-1/factor),size/2*(1-1/factor), factor))
def humanize_filesize(size):
	if size==0: return (0,"B")
	suffixes=['B','KiB','MiB','GiB','TiB','PiB','EiB','ZiB','YiB']
	
	po1024 = int(math.floor(math.log(size, 1024)))
	
	return (size/(1024**po1024), suffixes[po1024])


class FresnelKernel(Kernel):
	ndims = 3
	dtype=numpy.float64
	itype=numpy.int32
	
	def __init__(self, z,y,x,t,e, k, focus=0, mask=None, init=None):
		if init is None:
			self.__dict__.update(dict(z=z, y=y, x=x, t=t, e=e, k=k, focus=focus))

			self.idz = self.estimate_idz_range()
			self.dz = (self.z[1]-self.z[0])*self.idz

			self.shape = self.idz.shape+self.t.shape+self.e.shape+self.idz.shape+self.y.shape+self.x.shape
			self.fshape = (self.idz.size*self.t.size*self.e.size, self.idz.size*self.y.size*self.x.size)

			if mask:
				self.mask = numpy.add.outer((numpy.arange(y.size)-y.size//2)**2,(numpy.arange(x.size)-x.size//2)**2).flatten()<(.25*min(y.size**2, x.size**2))
			else:
				self.mask = numpy.ones(y.size*x.size, numpy.bool)
		else:
			self.__dict__.update(init)


	def estimate_idz_range(self):
		m_w = max(numpy.amax(self.x)-self.focus,-numpy.amin(self.x)+self.focus,numpy.amax(self.y)-self.focus,-numpy.amin(self.y)+self.focus)
		m_dz = int(numpy.ceil(numpy.sqrt(numpy.pi*2*m_w/self.k)/(self.z[1]-self.z[0])))
		return numpy.arange(-m_dz,m_dz+1, dtype=self.itype)

	def calc(self):
		e_idx, yx_idx = numpy.mgrid[:self.e.size,:self.y.size*self.x.size].astype(self.itype)
		e_idx = e_idx.flatten()#[:,self.mask].flatten()
		yx_idx = yx_idx.flatten()#[:,self.mask].flatten()

		#sel = numpy.empty(self.e.size*numpy.count_nonzero(self.mask), dtype=numpy.bool)
		sel = numpy.empty(self.e.size*self.y.size*self.x.size, dtype=numpy.bool)

		wvs = []

		for it, ti in Progress(enumerate(self.t), self.t.size):
			trafo = CT.Trafo2D(rad=ti)
			w,v,o = trafo.apply_to_bases(self.y,self.x)
			w,v = numexpr.evaluate("w+v+o", local_dict=dict(w=w[:,:,None],v=v[:,None,:],o=o[:,None,None])).reshape(2, self.y.size*self.x.size)
			w -= self.focus
			wvs.append((w,v))

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
				w,v = wvs[it]

                #e = self.e.repeat(w.size)
				#w = w[None,:].repeat(self.e.size, 0).flatten()
				#v = v[None,:].repeat(self.e.size, 0).flatten()

				sel = numexpr.evaluate('pi>=((v-e)**2-dv**2+z**2)*k/(2*abs(w))', local_dict=dict(z=zi, e=self.e[:,None], w=w[None,:], v=v[None,:], pi=numpy.pi, k=self.k, dw=dyx, dv=dyx)).flatten()

				if numpy.count_nonzero(sel)>0:

					e_sel = e_idx[sel]
					yx_sel = yx_idx[sel]

					dat = numexpr.evaluate("k/(2*pi*abs(w))*exp(j*k/(2*abs(w))*((v-e)**2+z**2))*dA",local_dict=dict(z=zi, v=v[yx_sel], w=w[yx_sel], e=self.e[e_sel], pi=numpy.pi, j=1j, k=self.k, dA=dA)).real

					psel = dat>0

					idat_z.append(numpy.require(dat[psel], requirements='O'))
					ite_z.append(numpy.require(it*self.e.size+e_sel[psel], requirements='O'))
					iyx_z.append(numpy.require(yx_sel[psel], requirements='O'))


			dat_z.append(MA.memconcatenate(idat_z, dtype=self.dtype))
			del idat_z
			te_z.append(MA.memconcatenate(ite_z, dtype=self.itype))
			del ite_z
			yx_z.append(MA.memconcatenate(iyx_z, dtype=self.itype))
			del iyx_z

		del dat, sel, trafo, e_idx, yx_idx, w,v,o

		self.bounds = [0]+[i.size for i in dat_z]
		self.bounds = numpy.cumsum(self.bounds).astype(self.itype)

		self.dat = numpy.concatenate(dat_z)
		self.idx_te = numpy.concatenate(te_z)
		self.idx_yx = numpy.concatenate(yx_z)

		self.idx = (self.idx_te, self.idx_yx)
		
	def save(self, filename):
		filename = os.path.expanduser(filename)
		
		hfile = h5py.File(filename, mode='w', libver='latest')
		for s in ('z','y','x','t','e','mask','dat','idx_te','idx_yx'):
			hfile.create_dataset(s, data=self.__dict__[s])
			
		stripped_self = self.__class__(0,0,0,0,0,0, init=dict(k=self.k, focus=self.focus, dtype=self.dtype, itype=self.itype, idz=self.idz, dz=self.dz, shape=self.shape, fshape=self.fshape, bounds=self.bounds))
		pickled_self = cPickle.dumps(stripped_self)
		
		hfile.attrs['self'] = numpy.void(pickled_self)
		
		hfile.flush()
		hfile.close()
		
		print("written: {:>8g}{:s}".format(*humanize_filesize(os.path.getsize(filename))))
		
	@staticmethod
	def load(filename):
		filename = os.path.expanduser(filename)
		
		hfile = h5py.File(filename, mode='r')
		self = cPickle.loads(str(hfile.attrs['self']))
		
		for s in ('z','y','x','t','e','mask','dat','idx_te','idx_yx'):
			self.__dict__[s] = numpy.empty(hfile[s].shape, hfile[s].dtype)
			hfile[s].read_direct(self.__dict__[s])
		self.idx = (self.idx_te, self.idx_yx)
		
		hfile.close()
		return self
