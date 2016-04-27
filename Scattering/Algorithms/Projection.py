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
from ..Operators.TransmissionFunctions import FlatAtomDW
from ..Potentials.AtomPotentials import WeickenmeierKohl
from ...Mathematics import FourierTransforms as FT
from ...Utilities import Physics
from ...Utilities import Progress

class Projection:
	def __init__(self, y, x, potential, energy, zi=None, zf=None, trafo=None, forgetful=False,
	             atom_potential_generator=WeickenmeierKohl, transmission_function=FlatAtomDW, transmission_function_args=None):

		if transmission_function_args is None: transmission_function_args = {}

		self.__dict__.update(dict(y=y, x=x, energy=energy,
		                          zi=zi, zf=zf, trafo=trafo,
		                          forgetful = forgetful,
		                          atom_potential_generator=atom_potential_generator, transmission_function=transmission_function, transmission_function_args=transmission_function_args))
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
		self.potential.zsort()

		if self.zf is None:
			self.zf = self.potential.zmax()
		if self.zi is None:
			self.zi = self.potential.zmin()
		
	def prepare(self):
		self.potential.zsort()

		self.opchain = OperatorChain(zi=self.zi, zf=self.zf)

		self.k = Physics.wavenumber(self.energy)

		self.kx, self.ky = FT.reciprocal_coords(self.x, self.y)
		self.kk =  numpy.add.outer(self.kx**2, self.ky**2)
		
		#for i in range(self.potential.atoms.size):
		self.opchain.append(self.transmission_function.inherit(self, self.potential.atoms))#[i:i+1]))

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

		if hasattr(wave, 'thread'):
			thread = wave.thread
			wave = wave.get()
			thread.synchronize()
			thread.release()
			del thread

		return wave
