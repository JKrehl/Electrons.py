from __future__ import absolute_import, division

import numpy
from ..Operators import OperatorChain
from ..Operators.TransferFunctions import FlatAtomDW
from ..Operators.Propagators import FresnelFourier
from ..Potentials.AtomPotentials import Kirkland
from ...Utilities import FourierTransforms as FT, Progress, Physics

class Multislice:
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

		if self.zf is None:
			self.zf = self.potential.zmax()
		if self.zi is None:
			self.zi = self.potential.zmin()
		
	def prepare(self):
		self.potential.zsort()
		
		self.opchain = OperatorChain(zi=self.zi, zf=self.zf)

		phaseshifts_f = {Z:self.atom_potential_generator.phaseshift_f(Z, self.energy, self.x, self.y) for Z in numpy.unique(self.potential.atoms['Z'])}

		kx, ky = FT.reciprocal_coords(self.x, self.y)
		kk =  numpy.add.outer(kx**2, ky**2)

		i = 0
		slice_thickness = Physics.wavenumber(self.energy)/(4*max(numpy.amax(numpy.abs(ky)), numpy.amax(numpy.abs(kx)))**2)
		while i<self.potential.atoms.size:
			j = i+1
			zi = self.potential.atoms['xyz'][i,2]
			while j<self.potential.atoms.size and self.potential.atoms['xyz'][j,2]<zi+slice_thickness:
				j += 1
						
			self.opchain.append(self.transfer_function(self.x, self.y, self.potential.atoms[i:j], kx=kx, ky=ky, kk=kk, phaseshifts_f=phaseshifts_f, lazy=True, forgetful=self.forgetful))
			i=j
			
		#print('slices:',self.opchain.size,'atoms:',self.potential.atoms.size, 'slice thickness:', slice_thickness)

		for zi, zf in self.opchain.get_gaps():
			self.opchain.append(self.propagator(zi,zf, self.energy, kx, ky, kk))

		self.opchain.impose_zorder()
		
		self.prepared = True
		
	def run(self, wave=None, progress=False):
		if wave is None:
			wave = numpy.ones(self.x.shape+self.y.shape, dtype=numpy.complex)
	
		if not self.prepared:
			self.prepare()

		if progress:
			for op in Progress(self.opchain['operator'], self.opchain.size):
				wave = op.apply(wave)
		else:
			for op in self.opchain['operator']:
				wave = op.apply(wave)

		return wave
