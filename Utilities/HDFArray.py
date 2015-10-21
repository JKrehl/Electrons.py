from __future__ import division, print_function, absolute_import

import numpy
import os.path
import tempfile
import h5py
import operator
import functools
import contextlib

class HDFConcatenator:
	def __init__(self, dtype=None):
		self.dtype = dtype
		
		self.tfile = tempfile.NamedTemporaryFile(dir=os.path.expanduser("~/tmp"))
		self.initialised = False

		self.sizes = []

	def append(self, arr):
		self.sizes.append(arr.size)
		
		if not self.initialised:
			if self.dtype is None:
				self.dtype = arr.dtype
			with h5py.File(self.tfile.name, 'r+') as hfile:
					hfile.create_dataset('array', data=arr.astype(self.dtype, copy=False), maxshape=(None,)+arr.shape[1:], chunks=True)
			self.initialised = True
		else:
			if arr.size!=0:
				with h5py.File(self.tfile.name, 'r+') as hfile:
					hfile['array'].resize(hfile['array'].shape[0]+arr.shape[0], axis=0)
					hfile['array'][-arr.shape[0]:, ...] = arr.astype(self.dtype, copy=False)

	@contextlib.contextmanager
	def array(self):
		with h5py.File(self.tfile.name, 'r+') as hfile:
			yield hfile['array']
