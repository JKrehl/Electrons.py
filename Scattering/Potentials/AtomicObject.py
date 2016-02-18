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
import os.path

from .load_cry import load_cry

class Atoms(numpy.ndarray):
	def __new__(cls, *args, **kwargs):
		if len(args)==0:
			args = [0]
		return numpy.ndarray.__new__(cls, *args, dtype={'names':('Z','zyx','B','occ'),'formats':(numpy.int, '3f' ,numpy.float, numpy.float)}, **kwargs)

	# noinspection PyUnusedLocal,PyUnusedLocal
	def __init__(self, *args, **kwargs):
		self['Z'] = 0
		self['zyx'] = 0
		self['B'] = 0
		self['occ'] = 1

	def __array_finalize__(self, obj):
		pass

	def append(self, Z, zyx, B=0, occ=1):
		self.resize(self.size+1, refcheck=False)
		self[-1] = (Z,zyx,B,occ)

class AtomicObject:
	def __init__(self, atoms=None, file=None, trafo=None):
		self.atoms = Atoms(0)

		if atoms is not None:
			self.atoms = atoms.copy()

		elif isinstance(file, str):
			file = os.path.expanduser(file)
			ext = os.path.splitext(file)[1]

			if ext == '.cry':
				data = load_cry(file)
				self.atoms = numpy.zeros(data['atoms'].shape, self.atoms.dtype)
				self.atoms['Z'] = data['atoms']['Z']
				self.atoms['zyx'] = data['atoms']['zyx']
				self.atoms['B'] = data['atoms']['B']
				self.atoms['occ'] = data['atoms']['occ']
				self.extent = data['extent']
				self.atoms = self.atoms[numpy.argsort(self.atoms['zyx'][:,0])]
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

		if trafo is not None:
			self.transform(trafo)


	def transform(self, trafo):
		atoms = self.atoms.copy()
		atoms['zyx'] = trafo*atoms['zyx']
		
		return self.__class__(atoms=atoms)

	def zsort(self):
		self.atoms[...] = self.atoms[numpy.argsort(self.atoms['zyx'][:,0])]

	def zmin(self):
		if self.atoms.size==0:
			return 0
		else:
			return numpy.amin(self.atoms['zyx'][:,0])
	
	def zmax(self):
		if self.atoms.size==0:
			return 0
		else:
			return numpy.amax(self.atoms['zyx'][:,0])

	def copy(self):
		return self.__class__(atoms=self.atoms.copy())
