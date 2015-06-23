from __future__ import division, print_function, absolute_import

import numpy
import os.path
import tempfile
import h5py
import operator

class HDFConcatenator:
	def __init__(self, dtype=None):
		self.dtype = dtype
		
		self.tfile = tempfile.NamedTemporaryFile(dir=os.path.expanduser("~/tmp"))
		self.hfile = h5py.File(self.tfile.name, 'r+')
		self.i = 0

	def append(self, arr, dtype=None):
		arr = numpy.require(arr)
		if self.dtype is None:
			if dtype is None:
				dtype = arr.dtype
		self.hfile.create_dataset(str(self.i), data=arr, dtype=self.dtype)
		self.i += 1

	@property
	def sizes(self):
		return [reduce(operator.mul, self.hfile[str(i)].shape) for i in xrange(self.i)]
	
	def concatenate(self):
		size = (reduce(operator.add, [reduce(operator.mul, i.shape) for i in self.hfile.values()]),)
		arr = numpy.empty(size, self.dtype)

		lb = 0
		for i in xrange(self.i):
			isize = reduce(operator.mul, self.hfile[str(i)].shape)
			if isize>0:
				self.hfile[str(i)].read_direct(arr, dest_sel=numpy.s_[lb:lb+isize])
				lb += isize

		return arr
