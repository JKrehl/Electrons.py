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

import h5py
import os.path

import tempfile
from contextlib import contextmanager
import pickle

from ...Utilities.Magic import humanize_filesize, MemberAccessDictionary

class Item(object):
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
		def __init__(self, parent, shape=(None,), dtype=None):
			self.parent = parent
			self.shape = shape
			self.dtype = dtype

		def append(self, value): raise NotImplementedError
		def finalize(self): raise NotImplementedError

	def concatenator(self, dtype=None, shape=(None,)):
		return self.Concatenator(self, dtype=dtype, shape=shape)

	def save(self): raise NotImplementedError
	@classmethod
	def load(cls, parent, name, path=None): raise NotImplementedError
		
class ArrayNDArray(Item):
	def __init__(self, parent, name):
		super().__init__(parent, name)
		self.ndarray = None
		
	@property
	def shape(self):
		return self.ndarray.shape
	@property
	def dtype(self):
		return self.ndarray.dtype
	
	def get(self):
		return self.ndarray
	def set(self, value):
		self.ndarray = value
		
	@contextmanager
	def in_memory(self):
		yield
		
	class Concatenator(Item.Concatenator):
		def __init__(self, parent, shape=(None,), dtype=None):
			if dtype is None:
				dtype = parent.dtype

			super().__init__(parent, shape, dtype)
			self.stack = []

		def append(self, value):
			assert value.shape[1:] == self.shape[1:]
			self.stack.append(value.astype(self.dtype, copy=False))

		def finalize(self):
			self.parent.set(numpy.concatenate(self.stack, axis=0))

	@contextmanager
	def interpret_path(self, path=None, mode='a'):
		if path is None:
			with self.parent.open() as hfile:
				yield hfile
		elif isinstance(path, h5py.File):
			yield path
		else:
			with h5py.File(path, mode) as hfile:
				yield hfile

	def save(self, path=None):
		with self.interpret_path(path) as hfile:
			if self.name not in hfile:
				hfile.create_dataset(self.name, data=self.ndarray)
			else:
				if hfile[self.name].dtype == self.ndarray.dtype and hfile[self.name].shape == self.ndarray.shape:
					hfile[self.name][...] = self.ndarray
				else:
					hfile.__delitem__(self.name)
					hfile.create_dataset(self.name, data=self.ndarray)

	@classmethod
	def load(cls, parent, name, path=None):
		self = cls(parent, name)

		with self.interpret_path(path) as hfile:
			self.ndarray = numpy.require(hfile[self.name], None, 'O')
				
		return self
	

class ArrayDataset(Item):
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
		if self.name in self.parent._hfile:
			return self.parent._hfile[self.name]
		else:
			return None
	
	@dataset.setter
	def dataset(self, value):
		with self.parent.open():
			if self.dataset is None:
				self.create(self.name, data=value)
			else:
				try: 
					self.dataset[...] = value
				except ValueError:
					self.create(data=value)
				except TypeError:
					self.create(data=value)

	@property
	def shape(self):
		if self.ndarray is None:
			with self.parent.open():
				return self.dataset.shape
		else:
			return self.ndarray.shape

	@property
	def dtype(self):
		if self.ndarray is None:
			with self.parent.open():
				return self.dataset.dtype
		else:
			return self.ndarray.dtype
	
	def get(self):
		if self.ndarray is not None:
			return self.ndarray
		else:
			return self.dataset
	def set(self, value):
		if self.ndarray is not None:
			self.ndarray = value
		else:
			with self.parent.open():
				if self.dataset is None:
					self.create(data=value)
				else:
					self.dataset = value
	
	@contextmanager
	def in_memory(self):
		if self.ndarray is None:
			with self.parent.open():
				self.ndarray = self.dataset[...]
			yield
			self.ndarray = None
		else:
			yield
		
	class Concatenator(Item.Concatenator):
		def __init__(self, parent, shape=(None,), dtype=None):
			if dtype is None:
				dtype = parent.dtype

			super().__init__(parent, shape, dtype)
			parent.create((0,)+shape[1:], dtype, maxshape=shape, chunks=(2**18,))

		def append(self, value):
			assert value.shape[1:] == self.shape[1:]
			if value.shape[0]!=0:
				with self.parent.parent.open():
					self.parent.dataset.resize(self.parent.dataset.shape[0]+value.shape[0], axis=0)
					self.parent.dataset[-value.shape[0]:,...] = value.astype(self.parent.dataset.dtype, copy=False)

		def finalize(self):
			pass

	@contextmanager
	def interpret_path(self, path=None, mode='a'):
		if path is None or path==self.parent.path:
			with self.parent.open() as hfile:
				yield hfile
		elif isinstance(path, h5py.File):
			yield path
		else:
			with h5py.File(path, mode) as hfile:
				yield hfile

	def save(self, path=None):
		if path is None or path==self.parent.path:
			pass
		else:
			with self.interpret_path(path) as hfile:
				if hfile is not None:
					with self.parent.open():
						hfile.copy(self.dataset, self.name)

	@classmethod
	def load(cls, parent, name, path=None):
		self = cls(parent, name)

		if parent.path != path:
			with self.interpret_path(path) as hfile:
				if hfile is not None:
					with self.parent.open() as parent_hfile:
						parent_hfile.copy(hfile[name], name)

		return self

class Scalar(Item):
	def __init__(self, parent, name):
		super().__init__(parent, name)
		self.value = None

	def get(self):
		return self.value
	def set(self, value):
		self.value = value

	@contextmanager
	def interpret_path(self, path=None, mode='a'):
		if path is None:
			with self.parent.open() as hfile:
				yield hfile
		elif isinstance(path, h5py.File):
			yield path
		else:
			with h5py.File(path, mode) as hfile:
				yield hfile

	def save(self, path=None):
		with self.interpret_path(path) as hfile:
			hfile.attrs[self.name] = numpy.void(pickle.dumps(self.value))

	@classmethod
	def load(cls, parent, name, path=None):
		self = cls(parent, name)

		with self.interpret_path(path) as hfile:
			if self.name in hfile.attrs:
				self.value = pickle.loads(bytes(hfile.attrs[self.name]))

		return self

	

class Kernel(object):
	def _init_path(self, path):
		if isinstance(path, str):
			self.path = os.path.expanduser(path)
			if not os.path.exists(os.path.dirname(path)):
				os.makedirs(os.path.dirname(path))
			self.tempfile = None
			if not os.path.isfile(self.path):
				h5py.File(self.path, mode='x').close()
		else:
			self.tempfile = tempfile.NamedTemporaryFile(dir=os.path.expanduser("~/tmp"), suffix=".hdf")
			self.path = self.tempfile.name

	def init_empty_arrays(self):
		if not hasattr(self, 'arrays'):
			self.arrays = MemberAccessDictionary()

		for key, mode in self._arrays.items():
			if key not in self.arrays:
				if mode == 0:
					self.arrays[key] = ArrayNDArray(self, key)
				elif mode == 1:
					self.arrays[key] = ArrayDataset(self, key)
				elif mode == 2:
					self.arrays[key] = Scalar(self, key)
				else:
					raise ValueError

	def load_arrays(self, path):
		if not hasattr(self, 'arrays'):
			self.arrays = MemberAccessDictionary()

		for key, mode in self._arrays.items():
			if key not in self.arrays:
				if mode == 0:
					self.arrays[key] = ArrayNDArray.load(self, key, path)
				elif mode == 1:
					self.arrays[key] = ArrayDataset.load(self, key, path)
				elif mode == 2:
					self.arrays[key] = Scalar.load(self, key, path)
				else:
					raise ValueError


	def __init__(self, path=None, *args, **kwargs):
		self._hfile = None

		if isinstance(path, str):
			self.path = os.path.expanduser(path)
			if not os.path.exists(os.path.dirname(path)):
				os.makedirs(os.path.dirname(path))
			self.tempfile = None
			if not os.path.isfile(self.path):
				h5py.File(self.path, mode='x').close()
		else:
			self.tempfile = tempfile.NamedTemporaryFile(dir=os.path.expanduser("~/tmp"), suffix=".hdf")
			self.path = self.tempfile.name

	@property
	def isopen(self):
		return self._hfile is not None and self._hfile._id.valid
	
	@contextmanager
	def open(self, mode='r+', driver=None):
		if not self.isopen:
			with h5py.File(self.path, mode, driver) as self._hfile:
				yield self._hfile
		else:
			yield self._hfile

	@contextmanager
	def in_memory(self, *arrays):
		cms = tuple(self.arrays[ar].in_memory() for ar in arrays)
		for cm in cms: cm.__enter__()
		yield
		for cm in cms: cm.__exit__(None, None, None)

	def __getattr__(self, name):
		if name!='_arrays' and name in self._arrays:
			return self.arrays[name].get()
		else:
			raise AttributeError("'%s' object has no attribute '%s'"%(self.__class__, name))
	
	def __setattr__(self, name, value):
		if name!='_arrays' and name in self._arrays:
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
				for key, array in self.arrays.items():
					array.save(hfile)

			print("written: {:>8g}{:s}".format(*humanize_filesize(os.path.getsize(path))))

	@classmethod
	def load(cls, path, own_path=0, *args, **kwargs):
		path = os.path.expanduser(path)
		if own_path is None:
			pass
		elif own_path == 0:
			own_path = path
		else:
			own_path = os.path.expanduser(own_path)

		self = cls.__new__(cls)
		self.__init__(own_path, *args, **kwargs)
		self.load_arrays(path)

		return self