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
import numexpr

from ....Mathematics import FourierTransforms as FT
from ...AtomPotentials import WeickenmeierKohl

from ...Operators import PlaneOperator
from ...Operators import AbstractArray

class FlatAtomDW(PlaneOperator):
	def __init__(self, atoms,
	             z = None,
	             ky = None, kx=None, kk=None,
	             phaseshifts_tf = None,
	             y = None, x = None,
	             atom_potential = WeickenmeierKohl, energy = None,
	             dtype = numpy.complex,
	             lazy = True, forgetful = True, factory = False):

		super().__init__(None)

		self.atoms = atoms

		self.dtype = dtype
		self.lazy, self.forgetful = lazy, forgetful
		self.factory = factory

		self.ky, self.kx, self.kk = ky, kx, kk
		if self.ky is None: self.ky = FT.reciprocal_coords(y)
		if self.kx is None: self.kx = FT.reciprocal_coords(x)
		if self.kk is None: self.kk = numpy.add.outer(self.ky**2, self.kx**2)

		self.phaseshifts_tf = phaseshifts_tf
		if self.phaseshifts_tf is None:
			if self.lazy:
				self.atom_potential = atom_potential
				self.energy = energy
				self.y, self.x = y, x
			else:
				self.phaseshifts_tf = {i: atom_potential.cis_phaseshift_f(i, energy, y, x) for i in numpy.unique(self.atoms['Z'])}

		self.transmission_function = None
		if self.lazy == 0:
			self.transmission_function = self.generate_transmission_function()

		self.z = z
		if self.z is None:
			self.z = numpy.mean(self.atoms['zyx'][:,0])

	def derive(self, atoms, **kwargs):
		args = dict(ky = self.ky, kx = self.kx, kk = self.kk,
		            phaseshifts_tf = self.phaseshifts_tf,
		            dtype = self.dtype,
		            lazy = self. lazy, forgetful = self.forgetful, factory = False)

		if hasattr(self, "atom_potential"):
			args.update(atom_potential=self.atom_potential, energy=self.energy, y=self.y, x=self.x)

		args.update(kwargs)

		return self.__class__(atoms, **args)

	def generate_transmission_function(self):
		if self.factory: return None

		if self.phaseshifts_tf is None:
			phaseshifts_tf = {i: self.atom_potential.cis_phaseshift_f(i, self.energy, self.y, self.x) for i in numpy.unique(self.atoms['Z'])}
			if not self.forgetful:
				self.phaseshifts_tf = phaseshifts_tf
		else:
			phaseshifts_tf = self.phaseshifts_tf

		transmission_function = numpy.ones(self.kk.shape, dtype=self.dtype)

		for a in self.atoms:
			transmission_function *= FT.ifft(numexpr.evaluate('ps*exp(-1j*(xs*kx+ys*ky)-kk*B/8)',
								   local_dict={'ps':phaseshifts_tf[a['Z']],
											   'ys':a['zyx'][1], 'xs':a['zyx'][2],
											   'ky':self.ky[:,None], 'kx':self.kx[None,:],
											   'kk':self.kk, 'B':a['B']/(4*numpy.pi**2)}))

		return transmission_function

	def apply(self, wave):
		if self.transmission_function is None:
			self.transmission_function = self.generate_transmission_function()

		if isinstance(wave, AbstractArray):
			wave = wave._as("numpy")

			numexpr.evaluate("tf*wave", local_dict=dict(tf=self.transmission_function, wave=wave), out=wave)

		if self.forgetful:
			self.transmission_function = None

		return wave
