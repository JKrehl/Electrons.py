 
from __future__ import division, print_function

import numpy
import numexpr
import scipy.interpolate

from ....Utilities import FourierTransforms as FT
from ...Potentials.AtomPotentials import Kirkland

from ..Base import PlaneOperator

def segment(a, keys=None):
    a = numpy.require(a)
    
    if keys is not None:
        unique, inverse = numpy.unique(keys, return_inverse=True)
    else:
        unique, inverse = numpy.unique(a, return_inverse=True)
    
        
    return tuple(a[inverse==i] for i in xrange(len(unique)))

class FlatAtomDW(PlaneOperator):
	def __init__(self, y, x, atoms, phaseshifts_f=None, roi=None, yoi=None, xoi=None, kx=None, ky=None, kk=None, z=None, atom_potential_gen=Kirkland, energy=None, lazy=False, forgetful=False):
		self.__dict__.update(dict(x=x, y=y, atoms=atoms, z=z,
								  roi=roi, yoi=yoi, xoi=xoi,
								  phaseshifts_f=phaseshifts_f, atom_potential_gen=atom_potential_gen, energy=energy,
								  kx=kx, ky=ky, kk=kk, lazy=lazy, forgetful=forgetful))

		if roi is None and yoi is None and xoi is None:
			self.patch = False
		else:
			self.patch = True
		
		self.transfer_function = None
		if not self.lazy:
			self.generate_tf()

		self.z = numpy.mean(self.atoms['zyx'][:,0])

	def generate_tf(self):
		if not self.patch:
			
			if self.phaseshifts_f is None:
				self.phaseshifts_f = {i: self.atom_potential_gen.phaseshift_f(i, self.energy, self.x, self.y) for i in numpy.unique(self.atoms['Z'])}
			
			if self.kx is None or self.ky is None:
				self.kx, self.ky = FT.reciprocal_coords(self.x, self.y)
			
			if self.kk is None:
				self.kk =  numpy.add.outer(self.kx**2, self.ky**2)
				
			tf = numpy.ones(self.kk.shape, dtype=numpy.complex)
		
			for a in self.atoms:
				tf += numexpr.evaluate('ps*exp(1j*(xs*kx+ys*ky)-kk*B/8)',
									   local_dict={'ps':self.phaseshifts_f[a['Z']],
												   'xs':a['zyx'][2],'ys':a['zyx'][1],
												   'kx':self.kx[:,None], 'ky':self.ky[None,:],
												   'kk':self.kk, 'B':a['B']})

			self.transfer_function = numpy.exp(1j*FT.ifft(tf))
			
		else:
			dy = self.y[1]-self.y[0]
			dx = self.x[1]-self.x[0]
			
			
			if self.yoi is None:
				liyoi = numpy.ceil(self.roi/dy)
				self.yoi = dy*numpy.arange(-liyoi, liyoi+1)
			else:
				liyoi = (self.yoi.size-1)//2
			if self.xoi is None:
				lixoi = numpy.ceil(self.roi/dx)
				self.xoi = dx*numpy.arange(-lixoi, lixoi+1)
			else:
				lixoi = (self.xoi.size-1)//2
			
			
		
			if self.phaseshifts_f is None:
				self.phaseshifts_f = {i: self.atom_potential_gen.phaseshift_f(i, self.energy, self.xoi, self.yoi) for i in numpy.unique(self.atoms['Z'])}
			
			if self.kx is None or self.ky is None:
				self.kx, self.ky = FT.reciprocal_coords(self.xoi, self.yoi)
		
			if self.kk is None:
				self.kk =  numpy.add.outer(self.kx**2, self.ky**2)

			tf = numpy.ones(self.y.shape+self.x.shape, dtype=numpy.complex)
		
			for a in self.atoms:
				py, px = a['zyx'][1], a['zyx'][2]
				rpy, ipy = numpy.modf((py-self.y[0])/dy)
				rpx, ipx = numpy.modf((px-self.x[0])/dx)

				itf = numexpr.evaluate('ps*exp(1j*(xs*kx+ys*ky)-kk*B/8)',
									   local_dict={'ps':self.phaseshifts_f[a['Z']],
												   'xs':rpx*dx,'ys':rpx*dy,
												   'kx':self.kx[:,None], 'ky':self.ky[None,:],
												   'kk':self.kk, 'B':a['B']})

				sl = numpy.s_[ipy-liyoi if ipy-liyoi>=0 else 0:ipy+liyoi+1 if ipy+liyoi+1<=self.y.size else self.y.size,
							  ipx-lixoi if ipx-lixoi>=0 else 0:ipx+lixoi+1 if ipx+lixoi+1<=self.x.size else self.x.size]
				isl = numpy.s_[0 if ipy-liyoi>=0 else liyoi-ipy:self.yoi.size if ipy+liyoi+1<=self.y.size else self.y.size-(ipy+liyoi+1),
							   0 if ipy-liyoi>=0 else liyoi-ipy:self.yoi.size if ipy+liyoi+1<=self.y.size else self.y.size-(ipy+liyoi+1)]
				
				tf[sl] *= numpy.exp(1j*FT.ifft(itf))[isl]
				

			self.transfer_function = tf

	def apply(self, wave):
		if self.transfer_function is None:
			self.generate_tf()
		res = self.transfer_function*wave
		if self.forgetful:
			self.transfer_function = None
		return res
