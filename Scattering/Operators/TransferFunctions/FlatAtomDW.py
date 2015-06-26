 
from __future__ import division, print_function

import numpy
import numexpr
from ....Utilities import FourierTransforms as FT
from ...Potentials.AtomPotentials import Kirkland

from ..Base import PlaneOperator

class FlatAtomDW(PlaneOperator):
	def __init__(self, x, y, atoms, phaseshifts_f=None, kx=None, ky=None, kk=None, z=None, atom_potential_gen=Kirkland, energy=None, lazy=False, forgetful=False):
		self.__dict__.update(dict(x=x, y=y, atoms=atoms, z=z,
								  phaseshifts_f=phaseshifts_f, atom_potential_gen=atom_potential_gen, energy=energy,
								  kx=kx, ky=ky, kk=kk, lazy=lazy, forgetful=forgetful))

		self.transfer_function = None
		if not self.lazy:
			self.generate_tf()

		self.z = numpy.mean(self.atoms['zyx'][:,0])

	def generate_tf(self):
		
		if self.phaseshifts_f is None:
			self.phaseshifts_f = {i: self.atom_potential_gen.phaseshift_f(i, self.energy, self.x, self.y) for i in numpy.unique(self.atoms['Z'])}
			
		if self.kx is None or self.ky is None:
			self.kx, self.ky = FT.reciprocal_coords(self.x, self.y)
		
		if self.kk is None:
			self.kk =  numpy.add.outer(self.kx**2, self.ky**2)

		tf = numpy.ones(self.kk.shape, dtype=numpy.complex)
		cis = lambda p:(numpy.cos(p)+1j*numpy.sin(p))

		for a in self.atoms:
			tf += numexpr.evaluate('ps*exp(1j*(xs*kx+ys*ky)-kk*B/8)',
								   local_dict={'ps':self.phaseshifts_f[a['Z']],
											   'xs':a['zyx'][2],'ys':a['zyx'][1],
											   'kx':self.kx[:,None], 'ky':self.ky[None,:],
											   'kk':self.kk, 'B':a['B']})

		#tf = numexpr.evaluate('sum(ps*exp(1j*(xs*kx+ys*ky)-kk*B/8), axis=0)',
		#								local_dict={'ps':self.phaseshifts_f[self.atoms[0]['Z']][None,:,:] ,
		#											'xs':self.atoms['xyz'][:,0][:,None,None],'ys':self.atoms['xyz'][:,1][:,None,None],
		#											'kx':self.kx[None,:,None], 'ky':self.ky[None,None,:],
		#											'kk':self.kk[None,:,:], 'B':self.atoms['B'][:,None,None]})
		
		self.transfer_function = numpy.exp(1j*FT.ifft(tf))

	def apply(self, wave):
		if self.transfer_function is None:
			self.generate_tf()
		res = self.transfer_function*wave
		if self.forgetful:
			self.transfer_function = None
		return res
