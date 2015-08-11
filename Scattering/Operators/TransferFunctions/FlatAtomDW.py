 
from __future__ import division, print_function

import numpy
import numexpr
import scipy.interpolate

from ....Mathematics import FourierTransforms as FT
from ...Potentials.AtomPotentials import WeickenmeierKohl

from ..Base import PlaneOperator

class FlatAtomDW(PlaneOperator):
	def __init__(self, atoms, phaseshifts_f=None,
				 ky=None, kx=None, kk=None,
				 atom_potential_generator=WeickenmeierKohl, energy=None, y=None, x=None,
				 dtype=numpy.complex,
				 lazy=True, forgetful=True):
		self.__dict__.update(dict(atoms=atoms,
								  phaseshifts_f=phaseshifts_f,
								  ky=ky, kx=kx, kk=kk,
								  atom_potential_generator=atom_potential_generator, energy=energy, y=y, x=x,
								  dtype=dtype,
								  lazy=lazy, forgetful=forgetful))
		
		self.transfer_function = None
		if not self.lazy:
			self.generate_tf()

		
		self.z = numpy.mean(self.atoms['zyx'][:,0])

	@staticmethod
	def ms_prep(parent):
		parent.phaseshifts_f = {i: parent.atom_potential_generator.phaseshift_f(i, parent.energy, parent.y, parent.x) for i in numpy.unique(parent.potential.atoms['Z'])}
	
	@classmethod
	def inherit(cls, parent, atoms, **kwargs):
		args = {}

		if hasattr(parent, 'kk'):
			args.update(dict(kk=parent.kk))
		if hasattr(parent, 'ky') and hasattr(parent, 'kx'):
			args.update(dict(ky=parent.ky, kx=parent.kx))
		if hasattr(parent, 'y') and hasattr(parent, 'x'):
			args.update(dict(y=parent.y, x=parent.x))
		

		if hasattr(parent, 'phaseshifts_f'):
			args.update(dict(phaseshifts_f=parent.phaseshifts_f))
		else:
			args.update(dict(energy=parent.energy, x=parent.x, y=parent.y))
			if hasattr(parent, 'atom_potential_generator'):
				args.update(dict(atom_potential_generator=parent.atom_potential_generator))

		args.update(kwargs)

		return cls(atoms, **args)
			
	def generate_tf(self):
		
		if self.phaseshifts_f is None:
			self.phaseshifts_f = {i: self.atom_potential_generator.phaseshift_f(i, self.energy, self.y, self.x) for i in numpy.unique(self.atoms['Z'])}

		if self.ky is None:
			ky = FT.reciprocal_coords(self.y)
		else:
			ky = self.ky
			
		if self.kx is None:
			kx = FT.reciprocal_coords(self.x)
		else:
			kx = self.kx

		if self.kk is None:
			kk = numpy.add.outer(ky**2, kx**2)
		else:
			kk = self.kk

		tf = numpy.zeros(kk.shape, dtype=self.dtype)

		for a in self.atoms:
			tf += numexpr.evaluate('ps*exp(1j*(xs*kx+ys*ky)-kk*B/8)',
								   local_dict={'ps':self.phaseshifts_f[a['Z']],
											   'xs':a['zyx'][2],'ys':a['zyx'][1],
											   'kx':kx[:,None], 'ky':ky[None,:],
											   'kk':kk, 'B':a['B']})

		self.transfer_function = numpy.exp(1j*FT.ifft(tf))

	def apply(self, wave):
		if self.transfer_function is None:
			self.generate_tf()
		res = numexpr.evaluate("tf*wave", local_dict=dict(tf=self.transfer_function, wave=wave))
		if self.forgetful:
			self.transfer_function = None
		return res
