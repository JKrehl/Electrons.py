import numpy

import h5py
import os.path
import pickle
import copy

from ...Utilities.Magic import humanize_filesize
from ...Utilities.Concatenator import NumpyConcatenator

class Kernel(object):
	_arrays = []

	def __init__(self):
		for key in self._arrays: setattr(self, key, None)

	def concatenator(self, key, dtype, shape):
		if key in self._arrays:
			return NumpyConcatenator(key, self, dtype, shape)
		else:
			return None
	
	def save(self, path):
		path = os.path.expanduser(path)

		bare_self = copy.copy(self)

		with h5py.File(path, mode='w') as hfile:

			for key in self._arrays:
				if getattr(self, key) is not None:
					hfile.create_dataset(key, data=getattr(self, key))
					setattr(bare_self, key, None)

			bare_self = numpy.void(pickle.dumps(bare_self))
			print("{:>16s}{:>8g}{:s}".format("size of bare self: ", *humanize_filesize(bare_self.nbytes)))
			hfile.attrs['self'] = bare_self
				
		print("{:>16s}{:>8g}{:s}".format("total filesize: ", *humanize_filesize(os.path.getsize(path))))

	@classmethod
	def load(cls, path):
		with h5py.File(path, mode='r') as hfile:
			self = pickle.loads(bytes(hfile.attrs['self']))

			for key in self._arrays:
				if key in hfile:
					setattr(self, key, numpy.require(hfile[key]))
				else:
					setattr(self, key, None)

		return self
