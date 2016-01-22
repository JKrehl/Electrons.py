#cython: boundscheck=False, initializedcheck=False, wraparound=False, initializedcheck=False

import numpy
cimport numpy
numpy.import_array()

from .Projector_Utilities cimport itype, dtype

def CS_pointers(numpy.ndarray[itype, ndim=1] vec, numpy.npy_intp size):
	cdef numpy.ndarray[itype, ndim=1] pointers = numpy.zeros(size+1, vec.dtype)

	pointers[0] = 0

	cdef numpy.npy_intp i, pi

	for i in range(vec.size):
		pointers[vec[i]+1] += 1

	pi = 0
	for i in range(size+1):
		pi += pointers[i]
		pointers[i] = pi

	return pointers