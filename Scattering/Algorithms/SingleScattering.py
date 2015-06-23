from __future__ import absolute_import, division

import numpy
from ..Operators import OperatorChain
from ..Operators.TransferFunctions import FlatAtomDW
from ..Operators.Propagators import FresnelFourier
from ..Potentials.AtomPotentials import Kirkland
from ...Utilities import FourierTransforms as FT, Progress

class SingleScattering:
	def __init__(self, x, y, potential, energy, zi=None, zf=None, trafo=None, forgetful=False, atom_potential_generator=Kirkland, transfer_function=FlatAtomDW, propagator=FresnelFourier):
		self.__dict__.update(dict(x=x, y=y, energy=energy,
								  zi=zi, zf=zf, trafo=trafo,
								  forgetful = forgetful,
								  atom_potential_generator=atom_potential_generator, transfer_function=transfer_function, propagator=propagator))
		self.prepared = False
		
		if self.trafo is not None:
			self.potential = potential.transform(self.trafo)
		else:
			self.potential = potential.copy()
		self.potential.zsort()

		if self.zf is None:
			self.zf = self.potential.zmax()
		if self.zi is None:
			self.zi = self.potential.zmin()
		
	def prepare(self):
		self.operators = []

		phaseshifts_f = {Z:self.atom_potential_generator.phaseshift_f(Z, self.energy, self.x, self.y) for Z in numpy.unique(self.potential.atoms['Z'])}

		kx, ky = FT.reciprocal_coords(self.x, self.y)
		kk =  numpy.add.outer(kx**2, ky**2)
		
		for i in xrange(self.potential.atoms.size):
			self.operators.append((self.propagator(self.zi, self.potential.atoms['xyz'][i,2], self.energy, kx, ky, kk),
								   self.transfer_function(self.x, self.y, self.potential.atoms[i:i+1], kx=kx, ky=ky, kk=kk, phaseshifts_f=phaseshifts_f, lazy=True, forgetful=self.forgetful),
								   self.propagator(self.potential.atoms['xyz'][i,2], self.zf, self.energy, kx, ky, kk)))

		self.empty_operator = self.propagator(self.zi, self.zf, self.energy, kx, ky, kk)
			
		self.prepared = True
		
	def run(self, iwave=None, iwave_propagating=True, progress=False):
		if iwave is None:
			iwave = numpy.ones(self.x.shape+self.y.shape, dtype=numpy.complex)
			
		if not self.prepared:
			self.prepare()

		if iwave_propagating:
			wave = self.empty_operator.apply(iwave)
		else:
			wave = iwave.copy()
			
		if progress:
			for iprop, tf, propf in Progress(self.operators):
				if iwave_propagating:
					ilwave = iprop.apply(iwave)
					wave += propf.apply(tf.apply(ilwave)-ilwave)
				else:
					wave += propf.apply(tf.apply(iwave)-iwave)
		else:
			for iprop, tf, propf in self.operators:
				if iwave_propagating:
					ilwave = iprop.apply(iwave)
					wave += propf.apply(tf.apply(ilwave)-ilwave)
				else:
					wave += propf.apply(tf.apply(iwave)-iwave)

		return wave
