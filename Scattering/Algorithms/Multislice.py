#!/usr/bin/env python
"""
Copyright (c) 2015 Jonas Krehl <Jonas.Krehl@triebenberg.de>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

import numpy

from ..Operators import OperatorChain
from ..Operators.TransferFunctions import FlatAtomDW
from ..Operators.Propagators import FresnelFourier
from ..Potentials.AtomPotentials import WeickenmeierKohl
from ...Utilities import Progress, Physics
from ...Mathematics import FourierTransforms as FT

class Multislice:
	def __init__(self, y, x, potential, energy, zi=None, zf=None, trafo=None, atom_potential_generator=WeickenmeierKohl,
				 transfer_function=FlatAtomDW, transfer_function_args=None,
				 propagator=FresnelFourier, propagator_args=None):

		if transfer_function_args is None: transfer_function_args = {}
		if propagator_args is None: propagator_args = {}

		
		self.__dict__.update(dict(y=y, x=x, energy=energy,
								  zi=zi, zf=zf, trafo=trafo,
								  atom_potential_generator=atom_potential_generator,
								  transfer_function=transfer_function, transfer_function_args=transfer_function_args,
								  propagator=propagator, propagator_args=propagator_args))
		self.prepared = False
		self.opchain = None
		self.k = None
		self.kx = None
		self.ky = None
		self.kk = None
		
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

		self.k = Physics.wavenumber(self.energy)
		
		self.ky, self.kx = FT.reciprocal_coords(self.y, self.x)
		self.kk = numpy.add.outer(self.ky**2, self.kx**2)

		i = 0
		slice_thickness = numpy.pi/20*2*Physics.wavenumber(self.energy)/(max(numpy.pi/(self.y[1]-self.y[0]), numpy.pi/(self.x[1]-self.x[0]))**2)
		while i<self.potential.atoms.size:
			j = i+1
			zi = self.potential.atoms['zyx'][i,0]
			while j<self.potential.atoms.size and self.potential.atoms['zyx'][j,0]<zi+slice_thickness:
				j += 1

			self.opchain.append(self.transfer_function.inherit(self, self.potential.atoms[i:j]))
			i=j

		for zi, zf in self.opchain.get_gaps():
			self.opchain.append(self.propagator.inherit(self, zi, zf))

		beg, end = self.opchain.get_caps()
		if beg is not None:
			self.opchain.prepend(self.propagator.inherit(self, *beg))
		if end is not None:
			self.opchain.append(self.propagator.inherit(self, *end))
		
		self.prepared = True

		return self
		
	def run(self, wave=None, progress=False):
		if wave is None:
			wave = numpy.ones(self.y.shape+self.x.shape, dtype=numpy.complex)
	
		if not self.prepared:
			self.prepare()
	
		if progress:
			for op in Progress(self.opchain['operator'], self.opchain.size):
				wave = op.apply(wave)
		else:
			for op in self.opchain['operator']:
				wave = op.apply(wave)

		if hasattr(wave, 'thread'):
			thread = wave.thread
			wave = wave.get()
			thread.synchronize()
			thread.release()
			del thread
			
		return wave
