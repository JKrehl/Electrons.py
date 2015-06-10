 
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

		self.z = self.atoms[0]['xyz'][2]

	def generate_tf(self):
		
		if self.phaseshifts_f is None:
			self.phaseshifts_f = {i: self.atom_potential_gen.phaseshift_f(i,self.energy,self.x,self.y) for i in numpy.unique(self.atoms['Z'])}
			
		if self.kx is None or self.ky is None:
			self.kx, self.ky = FT.reciprocal_coords(self.x, self.y)
		
		if self.kk is None:
			self.kk =  numpy.add.outer(self.kx**2, self.ky**2)

		tf = numpy.zeros(self.kk.shape, dtype=numpy.complex)
		cis = lambda p:(numpy.cos(p)+1j*numpy.sin(p))

		for a in self.atoms:
			tf += numexpr.evaluate('ps*exp(1j*(xs*kx+ys*ky)-kk*B/8)',
										local_dict={'ps':self.phaseshifts_f[a['Z']],
													'xs':a['xyz'][0],'ys':a['xyz'][1],
													'kx':self.kx[:,None], 'ky':self.ky[None,:],
													'kk':self.kk, 'B':a['B']})
		self.transfer_function = FT.ifft(tf)

	def apply(self, wave):
		if self.transfer_function is None:
			self.generate_tf()
		res = self.transfer_function*wave
		if self.forgetful:
			self.transfer_function = None
		return res
