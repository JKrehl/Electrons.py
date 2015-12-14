import numpy

import h5py
import os
import os.path
import pickle
import copy
from contextlib import contextmanager
import tempfile

from ...Utilities.Magic import humanize_filesize

class XDataset(h5py.Dataset):
	name = None
	
	def __init__(self, bind):
		super().__init__(self, bind)
		self.name = super().name
	
	def append(self, value, dtype):
		if dtype is None:
			dtype = value.dtype
			
			if key not in hfile:
				hfile.create_dataset(key, data=value.astype(dtype, copy=False), maxshape=(None,)+value.shape[1:], chunks=True)
			else:
				if value.size != 0:
					assert value.shape[1:]==hfile[key].shape[1:]

					self.resize(self.shape[0]+value.shape[0], axis=0)
					self.[-value.shape[0]:,...] = value.astype(hfile[key].dtype, copy=False)

	@classmethod
	def from_regular_dataset(cls, rd):
		self = cls(rd._id)
		return self

class Kernel:
	_default_in_memory = False
	_arrays = []
	_datasets = []
	_hfile = None
	
	def __init__(self, path=None, in_memory=_default_in_memory):
		if isinstance(path, str):
			self.path = os.path.expanduser(path)
			if not os.path.exists(os.path.dirname(path)):
				os.makedirs(os.path.dirname(path))
			self.tempfile = None
		else:
			if in_memory:
				self.tempfile = tempfile.NamedTemporaryFile(dir=os.path.expanduser("/tmp"), suffix=".hdf")
			else:
				self.tempfile = tempfile.NamedTemporaryFile(dir=os.path.expanduser("~/tmp"), suffix=".hdf")
			self.path = self.tempfile.name

		self.__dict__.update({key:None for key in self._arrays})
		self.__dict__.update({key:None for key in self._datasets})

	def sync(self):
		if self._hfile is not None:
			for key in self._datasets:
				if key in self._hfile:
					setattr(self, key, self._hfile[key])
		else:
			for key in self._datasets:
				setattr(self, key, None)
	
	@contextmanager
	def open(self, mode='r+'):
		if self._hfile is None:
			with h5py.File(self.path, mode) as self._hfile:
				self.sync()
				yield self._hfile
				self.sync()
		else:
			self.sync()
			yield self._hfile

	def create_append_dataset(self, key, dtype, maxshape):
		assert shape[0] is None

		with self.open() as hfile:
			hfile.create_dataset(key, dtype=dtype, shape=(0,)+maxshape[1:], maxshape=maxshape, chunks=True)
			
	def flush(self, key, overwrite=True):
		assert key in self._arrays
		value = getattr(self, key)
		
		with self.open() as hfile:
			if key in hfile and overwrite:
				if hfile[key].shape == value.shape and hfile[key].dtype == value.dtype:
					hfile[key][:] == value
				else:
					del hfile[key]
					hfile.create_dataset(key, data=value)
			else:
				hfile.create_dataset(key, data=value)

	def save(self, path = None):
		if path is None:
			path = self.path
		else:
			path = os.path.expanduser(path)

		bare_self = copy.copy(self)

		if path != self.path:
			with h5py.File(path, mode='w') as hfile:

				bare_self.path = path

				for key in self._arrays:
					if self.__dict__[key] is not None:
						hfile.create_dataset(key, data=self.__dict__[key])
						setattr(bare_self, key, None)

				with self.open() as ohfile:
					for key in self._datasets:
						if self.__dict__[key] is not None:
							ohfile.copy(key, hfile)
							setattr(bare_self, key, None)

				bare_self = numpy.void(pickle.dumps(bare_self))
				print("{:>16s}{:>8g}{:s}".format("size of bare self: ", *humanize_filesize(bare_self.nbytes)))
				hfile.attrs['self'] = bare_self
				
			print("{:>16s}{:>8g}{:s}".format("total filesize: ", *humanize_filesize(os.path.getsize(path))))
			
		else:
			with self.open() as hfile:
				for key in self._arrays:
					self.flush(key)
					setattr(bare_self, key, None)

				for key in self._datasets:
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
