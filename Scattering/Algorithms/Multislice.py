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
from ..Operators.Propagators import FresnelFourier
from ..AtomPotentials import WeickenmeierKohl
from ...Utilities import Progress, Physics
from ...Mathematics import FourierTransforms as FT

class Multislice:
	def __init__(self, y, x, specimen, energy, zi=None, zf=None, slicer="trivial", atom_potential_generator=WeickenmeierKohl,
	             transmission_function=FlatAtomDW, transmission_function_args={},
	             propagator=FresnelFourier, propagator_args={}):

		self.specimen = specimen
		self.slicer = slicer
		self.y, self.x = y, x
		self.energy = energy
		self.zi, self.zf = zi, zf
		self.atom_potential_generator = atom_potential_generator
		self.transmission_function = transmission_function
		self.transmission_function_args = transmission_function_args
		self.propagator = propagator
		self.propagator_args = propagator_args

		self.prepared = False
		self.opchain = None

		self.shared = dict()

	def prepare(self):
		self.opchain = OperatorChain(zi=self.zi, zf=self.zf)

		self.k = Physics.wavenumber(self.energy)
		
		self.ky, self.kx = FT.reciprocal_coords(self.y, self.x)
		self.kk = numpy.add.outer(self.ky**2, self.kx**2)

		self.transmission_function_factory = self.transmission_function(self.specimen.atoms,
		                                                                y=self.y, x=self.x, energy=self.energy,
		                                                                factory=True,
		                                                                **self.transmission_function_args)

		if self.slicer == "trivial":
			self.specimen.zsort()
			for i in range(self.specimen.atoms.size):
				self.opchain.append(self.transmission_function_factory.derive(self.specimen.atoms[i:i+1]))

		self.insert_propagators()
		
		self.prepared = True

		return self

	def insert_propagators(self):

		self.propagator_factory = self.propagator(0,0,
		                                          y = self.y, x = self.x, energy = self.energy,
		                                          factory=True,
		                                          **self.propagator_args)

		for i, zi, zf in self.opchain.get_gaps(indices=True)[::-1]:
			self.opchain.insert(i, self.propagator_factory.derive(zi, zf))

	def remove_propagators(self):
		self.opchain = self.opchain[self.opchain['zi'] == self.opchain['zf']].copy()

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
			
		return wave
