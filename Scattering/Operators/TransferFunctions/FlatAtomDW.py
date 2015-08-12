 
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

	@classmethod
	def inherit(cls, parent, atoms, **kwargs):
		args = {}

		args.update(parent.transfer_function_args)
		args.update(kwargs)
		args.update({s:parent.__dict__[s] for s in ['y', 'x', 'ky', 'kx', 'kk'] if s not in args or args[s] is None})

		if 'phaseshifts_f' not in args or args['phaseshifts_f'] is None:
			if hasattr(parent, 'phaseshifts_f') and parent.phaseshifts_f is not None:
				args['phaseshifts_f'] = parent.phaseshifts_f
			else:
				if 'energy' not in args or args['energy'] is None:
					args['energy'] = parent.energy
				if 'atom_potential_generator' not in args or args['atom_potential_generator'] is None:
					args['atom_potential_generator'] = parent.atom_potential_generator
				args['phaseshifts_f'] = {i: args['atom_potential_generator'].phaseshift_f(i, args['energy'], args['y'], args['x']) for i in numpy.unique(atoms['Z'])}
				
		parent.transfer_function_args.update(args)

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
			tf += numexpr.evaluate('ps*exp(-1j*(xs*kx+ys*ky)-kk*B/8)',
								   local_dict={'ps':self.phaseshifts_f[a['Z']],
											   'ys':a['zyx'][1], 'xs':a['zyx'][2],
											   'ky':ky[:,None], 'kx':kx[None,:],
											   'kk':kk, 'B':a['B']})

		self.transfer_function = numpy.exp(1j*FT.ifft(tf))

	def apply(self, wave):
		if self.transfer_function is None:
			self.generate_tf()
		res = numexpr.evaluate("tf*wave", local_dict=dict(tf=self.transfer_function, wave=wave))
		if self.forgetful:
			self.transfer_function = None
		return res
