from __future__ import print_function

import numpy
import scipy.sparse

import h5py
import copy

try:
	import cPickle
except ImportError:
	import pickle

import os.path
import math

def humanize_filesize(size):
	if size==0: return (0,"B")
	suffixes=['B','KiB','MiB','GiB','TiB','PiB','EiB','ZiB','YiB']
	
	po1024 = int(math.floor(math.log(size, 1024)))
	
	return (size/(1024**po1024), suffixes[po1024])

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
				
			hfile.attrs['self'] = numpy.void(cPickle.dumps(stripped_self))
			
		print("written: {:>8g}{:s}".format(*humanize_filesize(os.path.getsize(filename))))

	@classmethod
	def load(cls, filename):
		filename = os.path.expanduser(filename)

		with h5py.File(filename, mode='r') as hfile:
			self = cPickle.loads(str(hfile.attrs['self']))

			for name, value in hfile.iteritems():
				self.__dict__[name] = numpy.require(value)

		return self
