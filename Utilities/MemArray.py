from __future__ import division, print_function

import numpy
import tempfile
import os

class memarray(numpy.memmap):	
	def __new__(subtype,array=None, shape=None, dtype=None):
		if shape is None:
			shape = array.shape
		if dtype is None:
			dtype = array.dtype
		if sum(shape)==0:
			return numpy.ndarray.__new__(subtype, shape=shape, dtype=dtype)
		else:
			fle = tempfile.NamedTemporaryFile(dir=os.path.expanduser("~/tmp"))
			new =  numpy.memmap.__new__(subtype, fle, dtype, 'w+', 0, shape)
			new.file = fle
			return new

	def __init__(self, array=None, shape=None, dtype=None):
		if array is not None:
			self[...] = array[...]

def memconcatenate(arrs, dtype=None):
	if dtype is None:
		if len(arrs)>0:
			dtype = arrs[0].dtype
		else:
			dtype = numpy.float
			
	if len(arrs)==0:
		return numpy.array([], dtype=dtype)
	
	shape = (sum((i.size for i in arrs)),)
	re = memarray(shape=shape, dtype=dtype)
	i = 0
	for a in arrs:
		re[i:i+a.size] = a
		i += a.size
	return re
