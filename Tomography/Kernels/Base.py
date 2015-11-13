import numpy
import scipy.sparse

import h5py
import copy

import pickle

import os.path
import math

from ...Utilities.Magic import humanize_filesize

class Kernel:
	ndims = None
	fshape = None
	shape = None

	dat = None
	idx = None

	def as_coo(self):
		return scipy.sparse.coo_matrix((self.dat, (self.col, self.row)), shape=self.fshape)
	
	def __init__(self):
		pass

	def calc(self):
		raise NotImplemented

	def save(self, filename):
		filename = os.path.expanduser(filename)
		stripped_self = copy.copy(self)

		with h5py.File(filename, mode='w') as hfile:
			
			for name in self.__arrays__:
				hfile.create_dataset(name, data=self.__dict__[name])
				setattr(stripped_self, name, None)
				
			hfile.attrs['self'] = numpy.void(pickle.dumps(stripped_self))
			
		print("written: {:>8g}{:s}".format(*humanize_filesize(os.path.getsize(filename))))

	@classmethod
	def load(cls, filename):
		filename = os.path.expanduser(filename)

		with h5py.File(filename, mode='r') as hfile:
			self = pickle.loads(bytes(hfile.attrs['self']))

			for name, value in hfile.items():
				self.__dict__[name] = numpy.require(value)

		return self

class CSKernel:
	ndims = None
	fshape = None
	shape = None

	cc_dat = None
	cc_idx = None
	cc_ptr = None
	
	cr_dat = None
	cr_idx = None
	cc_ptr = None

	def __init__(self):
		pass


	@classmethod
	def from_coo(cls, kernel):
		pass
	
	def save(self, filename):
		filename = os.path.expanduser(filename)
		stripped_self = copy.copy(self)

		with h5py.File(filename, mode='w') as hfile:
			
			for name in self.__arrays__:
				hfile.create_dataset(name, data=self.__dict__[name])
				setattr(stripped_self, name, None)
				
			hfile.attrs['self'] = numpy.void(pickle.dumps(stripped_self))
			
		print("written: {:>8g}{:s}".format(*humanize_filesize(os.path.getsize(filename))))

	@classmethod
	def load(cls, filename):
		filename = os.path.expanduser(filename)

		with h5py.File(filename, mode='r') as hfile:
			self = pickle.loads(bytes(hfile.attrs['self']))

			for name, value in hfile.items():
				self.__dict__[name] = numpy.require(value)

		return self
