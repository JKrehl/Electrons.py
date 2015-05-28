#cython: boundscheck=False, initializedcheck=False, wraparound=False

import numpy
cimport numpy, cython

ctypedef fused idx_t:
	numpy.int32_t
	numpy.int64_t
	numpy.uint32_t
	numpy.int_t

ctypedef fused dat_t:
	numpy.float32_t
	numpy.float64_t
	numpy.float_t

def matvec(
		dat_t[:] vec,
		dat_t[:] res,
		dat_t[:] dat,
		idx_t[:] col,
		idx_t[:] row,
		):

	cdef idx_t tensor_length = dat.size
	cdef idx_t i

	with nogil:
		for i in range(tensor_length):
			res[row[i]] += dat[i]*vec[col[i]]


