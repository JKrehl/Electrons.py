import numpy

import h5py
import os.path

import tempfile
from contextlib import contextmanager

from ...Utilities.Magic import humanize_filesize, MemberAccessDictionary

class Array(object):
	def __init__(self, parent, name):
		self.parent = parent
		self.name = name
	
	@property
	def shape(self): return NotImplemented
	@property
	def dtype(self): return NotImplemented
	
	def get(self): return NotImplemented
	def set(self, value): return NotImplemented
	
	@contextmanager
	def in_memory(self): raise NotImplementedError
	
	class Concatenator(object):
		def __init__(self, parent, shape, dtype):
			self.parent = parent
			self.shape = shape
			self.dtype = dtype

		def append(self, value): raise NotImplementedError
		def finalize(self): raise NotImplementedError
	
	def get_concatenator(self, shape=(None,), dtype=None):
		if dtype is None:
			dtype = self.dtype

		return self.Concatenator(self, shape, dtype)
	
	def save(self): raise NotImplementedError
	def save_in_other(self, other): raise NotImplementedError
	@classmethod
	def load(cls, parent, name): raise NotImplementedError
		
class Array_NDArray(Array):
	def __init__(self, parent, name):
		super().__init__(parent, name)
		self.ndarray = None
		
	@property
	def shape(self): return self.ndarray.shape
	@property
	def dtype(self): return self.ndarray.dtype
	
	def get(self):
		return self.ndarray
	def set(self, value):
		self.ndarray = value
		
	@contextmanager
	def in_memory(self):
		yield
		
	class Concatenator(Array.Concatenator):
		def __init__(self, parent, shape, dtype):
			super().__init__(parent, shape, dtype)
			self.stack = []

		def append(self, value):
			assert value.shape[1:] == self.shape[1:]
			self.stack.append(value.astype(self.dtype, copy=False))

		def finalize(self):
			self.parent.set(numpy.concatenate(self.stack, axis=0))

	def save_in_other(self, other):
		if self.name not in other:
			other.create_dataset(self.name, data=self.ndarray)
		else:
			try: 
				other[self.name][...] = self.ndarray
			except ValueError:
				other.__delitem__(self.name)
				other.create_dataset(self.name, data=self.ndarray)
			
	def save(self):
		with self.parent.open() as hfile:
			self.save_in_other(hfile)

	@classmethod
	def load(cls, parent, name, path=None):
		self = cls(parent, name)
		
		if path is None:
			with self.parent.open() as hfile:
				self.ndarray = numpy.require(hfile[self.name], None, 'O')
		else:
			with h5py.File(path, 'r') as hfile:
				self.ndarray = numpy.require(hfile[self.name], None, 'O')
				
		return self
	

class Array_Dataset(Array):
	def __init__(self, parent, name):
		super().__init__(parent, name)
		self.ndarray = None
		
	def create(self, shape=None, dtype=None, data=None, **kwargs):
		with self.parent.open() as hfile:
			if self.name in hfile:
				hfile.__delitem__(self.name)
			hfile.create_dataset(self.name, shape=shape, dtype=dtype, data=data, **kwargs)
			
	@property
	def dataset(self):
		with self.parent.open() as hfile:
			if self.name in hfile:
				return hfile[self.name]
			else:
				return None
	
	@dataset.setter
	def dataset(self, value):
		with self.parent.open() as hfile:
			if self.dataset is None:
				self.create(self.name, data=value)
			else:
				try: 
					self.dataset[...] = value
				except ValueError:
					self.create(self.name, data=value)
	@property
	def shape(self):
		return self.ndarray is None and self.dataset.shape or self.ndarray.shape
		
	@property
	def dtype(self): 
		return self.ndarray is None and self.dataset.dtype or self.ndarray.dtype
	
	def get(self):
		if self.ndarray is not None:
			return self.ndarray
		else:
			return self.dataset
	def set(self, value):
		if self.ndarray is not None:
			self.ndarray = value
		else:
			if self.dataset is None:
				self.create(data=value)
			else:
				self.dataset = value
	
	@contextmanager
	def in_memory(self):
		self.ndarray = numpy.require(self.dataset)
		yield
		self.dataset = self.ndarray
		self.ndarray = None
		
	class Concatenator(Array.Concatenator):
		def __init__(self, parent, shape, dtype):
			super().__init__(parent, shape, dtype)
			parent.create((0,)+shape[1:], dtype, maxshape=shape, chunks=True)
			
			
		def append(self, value): raise NotImplementedError
		def finalize(self): raise NotImplementedError
	
	def get_concatenator(self, shape=(None,), dtype=None):
		if dtype is None:
			dtype = self.dtype

		return self.Concatenator(self, shape, dtype)

	def save(self):
		pass
	def save_in_other(self, other):
		with self.parent.open():
			other.copy(self.dataset, other)

	@classmethod
	def load(cls, parent, name, path=None):
		self = cls(parent, name)

		if path != parent.path:
			with h5py.File(path, 'r') as hfile:
				with self.parent.open() as hfile2:
					hfile2.copy(hfile[name], name)

		return self

class Kernel(object):
	_arrays = {}

	def infer_path(self, path):
		if isinstance(path, str):
			self.path = os.path.expanduser(path)
			if not os.path.exists(os.path.dirname(path)):
				os.makedirs(os.path.dirname(path))
			self.tempfile = None
		else:
			self.tempfile = tempfile.NamedTemporaryFile(dir=os.path.expanduser("~/tmp"), suffix=".hdf")
			self.path = self.tempfile.name
	
	def __init__(self, path=None):
		self._hfile = None

		self.infer_path(path)

		self.arrays = MemberAccessDictionary()
		for key, mode in self._arrays.items():
			if mode == 0:
				self.arrays[key] = Array_NDArray(self, key)
			elif mode == 1:
				self.arrays[key] = Array_Dataset(self, key)
			else:
				raise ValueError
	@property
	def isopen(self):
		return self._hfile is not None and self._hfile._id.valid
	
	@contextmanager
	def open(self, mode='a'):
		if not self.isopen:
			with h5py.File(self.path, mode) as self._hfile:
				yield self._hfile
		else:
			yield self._hfile
	
	def __getattr__(self, name):
		if name in self._arrays:
			return self.arrays[name].get()
		else:
			raise AttributeError("'%s' object has no attribute '%s'"%(self.__class__, name))
	
	def __setattr__(self, name, value):
		if name in self._arrays:
			self.arrays[name].set(value)
		else:
			object.__setattr__(self, name, value)
			
	def save(self, path=None):
		if path is not None:
			path = os.path.expanduser(path)
		
		if path is None or path==self.path:
			for array in self.arrays.values():
				array.save()

			print("written: {:>8g}{:s}".format(*humanize_filesize(os.path.getsize(self.path))))
		else:
			with h5py.File(path, 'w') as hfile:
				for array in self.arrays.values():
					array.save_in_other(hfile)

			print("written: {:>8g}{:s}".format(*humanize_filesize(os.path.getsize(path))))
		

	@classmethod
	def load(cls, path, own_path=0):
		path = os.path.expanduser(path)
		self = cls.__new__(cls)

		self._hfile = None
		
		if own_path == 0:
			own_path = path
			
		self.infer_path(own_path)

		self.arrays = MemberAccessDictionary()
		for key, mode in self._arrays.items():
			if mode == 0:
				self.arrays[key] = Array_NDArray.load(self, key, path)
			elif mode == 1:
				self.arrays[key] = Array_Dataset.load(self, key, path)
			else:
				raise ValueError
			
		return self
		
		
