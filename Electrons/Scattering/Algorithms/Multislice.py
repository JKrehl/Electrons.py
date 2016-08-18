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
import inspect

from ..Operators import OperatorChain
from ..Operators.TransmissionFunctions import FlatAtomDW
from ..Operators.Propagators import FresnelFourier
from ..AtomPotentials import WeickenmeierKohl
from ...Utilities import Progress, Physics
from ...Mathematics import FourierTransforms as FT
from ..Operators import AbstractArray

class Multislice:
	def __init__(self, y, x, specimen, energy, zi=None, zf=None, slicer="bunching", atom_potential_generator=WeickenmeierKohl,
	             transmission_function=FlatAtomDW, transmission_function_args=None,
	             propagator=FresnelFourier, propagator_args=None):

		self.specimen = specimen
		self.slicer = slicer
		self.y, self.x = y, x
		self.energy = energy
		self.zi, self.zf = zi, zf
		self.atom_potential_generator = atom_potential_generator
		self.transmission_function = transmission_function
		self.transmission_function_args = (lambda a: dict() if a is None else a)(transmission_function_args)
		self.propagator = propagator
		self.propagator_args = (lambda a: dict() if a is None else a)(propagator_args)

		self.prepared = False
		self.opchain = OperatorChain(zi=self.zi, zf=self.zf)

		self.shared = dict()

	def prepare(self):
		self.k = Physics.wavenumber(self.energy)

		self.ky, self.kx = FT.reciprocal_coords(self.y, self.x)
		self.kk = numpy.add.outer(self.ky**2, self.kx**2)

		self.transmission_function_generator = self.transmission_function(self.specimen.atoms,
		                                                                  y=self.y, x=self.x, energy=self.energy,
		                                                                  factory=True,
		                                                                  **self.transmission_function_args)

		if self.slicer == "trivial":
			self.specimen.zsort()
			for i in range(self.specimen.atoms.size):
				self.opchain.append(self.transmission_function_generator.derive(self.specimen.atoms[i:i + 1]))

		elif self.slicer == "bunching":
			slice_thickness = numpy.pi/20*2*Physics.wavenumber(self.energy)/(max(numpy.pi/(self.y[1]-self.y[0]), numpy.pi/(self.x[1]-self.x[0]))**2)
			zs = self.specimen.atoms['zyx'][:,0]
			mean_z = numpy.mean(zs)
			inslice = numpy.round((zs-mean_z)/slice_thickness)
			for iz in numpy.unique(inslice):
				self.opchain.append(self.transmission_function_generator.derive(self.specimen.atoms[inslice == iz], z =mean_z + iz * slice_thickness))
		else:
			raise NotImplementedError

		self.insert_propagators()

		self.prepared = True

		return self

	def insert_propagators(self):

		if hasattr(self.transmission_function_generator, "thread"):
			if "thread" not in self.propagator_args and "thread" in inspect.signature(self.propagator).parameters:
				self.propagator_args["thread"] = self.transmission_function_generator.thread
		if hasattr(self.transmission_function_generator, "mode"):
			if "mode" not in self.propagator_args and "mode" in inspect.signature(self.propagator).parameters:
				self.propagator_args["mode"] = self.transmission_function_generator.mode


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
			wave = AbstractArray(numpy.ones(self.y.shape+self.x.shape, dtype=numpy.complex))
		elif not issubclass(wave.__class__, numpy.ndarray):
			wave = AbstractArray(wave)

		if not self.prepared:
			self.prepare()

		if progress:
			for op in Progress(self.opchain['operator'], self.opchain.size, keep=True):
				wave = op.apply(wave)
		else:
			for op in self.opchain['operator']:
				wave = op.apply(wave)

		#wave = wave.set_mode("numpy")
		if isinstance(wave, AbstractArray):
			wave = wave.to_ndarray()
		return wave
