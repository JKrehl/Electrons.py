import numpy

import pyximport
pyximport.install()

from .CompressedSparse_cython import CS_pointers

def compress_sparse(vec, size, *arrays):
	pointers = CS_pointers(vec, size)

	srt = numpy.argsort(vec)

	return (pointers,) + tuple(arr[srt] for arr in arrays)