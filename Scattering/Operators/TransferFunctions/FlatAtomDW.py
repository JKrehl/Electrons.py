 
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

		if roi is not None and yoi is not None and xoi is not None:
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
		else:
			
			if self.yoi is None:
				liyoi = numpy.ceil(roi/(self.y[1]-self.y[0]))
				self.yoi = (self.y[1]-self.y[0])*numpy.arange(-liyoi, liyoi+1)
			else:
				liyoi = (self.yoi.size-1)//2
			if xoi is None:
				lixoi = numpy.ceil(roi/(self.x[1]-self.x[0]))
				self.xoi = (self.x[1]-self.x[0])*numpy.arange(-lixoi, lixoi+1)
			else:
				lixoi = (self.xoi.size-1)//2
			
			
		
			if self.phaseshifts_f is None:
				self.phaseshifts_f = {i: self.atom_potential_gen.phaseshift_f(i, self.energy, self.xoi, self.yoi) for i in numpy.unique(self.atoms['Z'])}
			
			if self.kx is None or self.ky is None:
				self.kx, self.ky = FT.reciprocal_coords(self.xoi, self.yoi)
		
			if self.kk is None:
				self.kk =  numpy.add.outer(self.kx**2, self.ky**2)

			tf = numpy.ones(self.kk.shape, dtype=numpy.complex)
		
			for a in self.atoms:
				py, px = a['zyx'][1],['zyx'][2]
				rpy, ipy = numpy.modf(py)
				rpx, ipx = numpy.modf(px)

				itf = numexpr.evaluate('ps*exp(1j*(xs*kx+ys*ky)-kk*B/8)',
									   local_dict={'ps':self.phaseshifts_f[a['Z']],
												   'xs':rpx,'ys':rpx,
												   'kx':self.kx[:,None], 'ky':self.ky[None,:],
												   'kk':self.kk, 'B':a['B']})

				initf = scipy.interpolate.RegularGridInterpolator((self.yoi+ipy, self.xoi+ipx), itf, fill_value=0)(
				

		#tf = numpy.ones(self.kk.shape, dtype=numpy.complex)
		#print(len(segment(self.atoms, self.atoms['Z'])), self.atoms.size, segment(self.atoms, self.atoms['Z'])[0].size)
		#for a in segment(self.atoms, self.atoms['Z']):
		#	tf += self.phaseshifts_f[a[0]['Z']]*numexpr.evaluate('sum(exp(1j*(xs*kx+ys*ky)-kk*B/8), axis=0)',
		#														 local_dict={'ys':a['zyx'][:,1][:,None,None],'xs':a['zyx'][:,2][:,None,None],
		#																	 'ky':self.ky[None,:,None], 'kx':self.kx[None,None,:],
		#																	 'kk':self.kk[None,:,:], 'B':a['B'][:,None,None]})
		
		self.transfer_function = numpy.exp(1j*FT.ifft(tf))

	def apply(self, wave):
		if self.transfer_function is None:
			self.generate_tf()
		res = self.transfer_function*wave
		if self.forgetful:
			self.transfer_function = None
		return res
