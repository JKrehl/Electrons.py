from __future__ import absolute_import, division

import numpy
from ..Operators import OperatorChain
from ..Operators.TransferFunctions import FlatAtomDW
from ..Operators.Propagators import FresnelFourier
from ..Potentials.AtomPotentials import WeickenmeierKohl
from ...Utilities import Progress, Physics
from ...Mathematics import FourierTransforms as FT

class Multislice:
	def __init__(self, x, y, potential, energy, zi=None, zf=None, trafo=None, forgetful=False, roi=None, atom_potential_generator=WeickenmeierKohl, transfer_function=FlatAtomDW, propagator=FresnelFourier):
		self.__dict__.update(dict(x=x, y=y, energy=energy,
								  zi=zi, zf=zf, trafo=trafo,
								  forgetful = forgetful, roi=roi,
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

		k = Physics.wavenumber(self.energy)
		
		transfer_function_prep = self.transfer_function.prep(self.y, self.x, self.potential.atoms, atom_potential_generator=self.atom_potential_generator, energy=self.energy, roi=self.roi, forgetful=self.forgetful, lazy=True)
		propagator_prep = self.propagator.prep(k, y=self.y, x=self.x)
		
		i = 0
		slice_thickness = Physics.wavenumber(self.energy)/(4*max(numpy.pi/(self.y[1]-self.y[0]), numpy.pi/(self.x[1]-self.x[0]))**2)
		while i<self.potential.atoms.size:
			j = i+1
			zi = self.potential.atoms['zyx'][i,0]
			while j<self.potential.atoms.size and self.potential.atoms['zyx'][j,0]<zi+slice_thickness:
				j += 1

			self.opchain.append(self.transfer_function(self.y, self.x, self.potential.atoms[i:j], **transfer_function_prep))
			
			i=j
			
		#print('slices:',self.opchain.size,'atoms:',self.potential.atoms.size, 'slice thickness:', slice_thickness)

		for zi, zf in self.opchain.get_gaps():
			self.opchain.append(self.propagator(zi,zf, **propagator_prep))

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
