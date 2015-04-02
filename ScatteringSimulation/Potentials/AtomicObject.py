from __future__ import absolute_import, division, print_function

import numpy
import os.path

from .load_cry import load_cry

class Atoms(numpy.ndarray):
	def __new__(cls, *args, **kwargs):
		if len(args)==0:
			args = [0]
		return numpy.ndarray.__new__(cls, *args, dtype={'names':('Z','xyz','B','occ'),'formats':(numpy.int, '3f' ,numpy.float, numpy.float)}, **kwargs)
	
	def __init__(self, *args, **kwargs):
		self['Z'] = 0
		self['xyz'] = 0
		self['B'] = 0
		self['occ'] = 1

	def __array_finalize__(self, obj):
		pass

	def append(self, Z, xyz, B=0, occ=1):
		self.resize(self.size+1, refcheck=False)
		self[-1] = (Z,xyz,B,occ)

class AtomicObject:
	def __init__(self, atoms=None, file=None, trafo=None):
		self.atoms = Atoms(0)

		if atoms is not None:
			self.atoms = atoms.copy()

		elif isinstance(file, basestring):
			file = os.path.expanduser(file)
			ext = os.path.splitext(file)[1]

			if ext == '.cry':
				data = load_cry(file)
				self.atoms = numpy.zeros(data['atoms'].shape, self.atoms.dtype)
				self.atoms['Z'] = data['atoms']['Z']
				self.atoms['xyz'] = data['atoms']['xyz']
				self.atoms['B'] = data['atoms']['B']
				self.atoms['occ'] = data['atoms']['occ']
				self.extent = data['extent']
				self.atoms = self.atoms[numpy.argsort(self.atoms['xyz'][:,2])]
			else:
				raise NotImplemented
		else:
			raise NotImplemented

		if trafo is not None:
			self.transform(trafo)


	def transform(self, trafo):
		atoms = self.atoms.copy()
		atoms['xyz'] = trafo*atoms['xyz']
					
		return self.__class__(atoms=atoms)

	def zsort(self):
		self.atoms[...] = self.atoms[numpy.argsort(self.atoms['xyz'][:,2])]

	def zmin(self):
		return numpy.amin(self.atoms['xyz'][:,2])
	
	def zmax(self):
		return numpy.amax(self.atoms['xyz'][:,2])

	def copy(self):
		return self.__class__(atoms=atoms.copy())
