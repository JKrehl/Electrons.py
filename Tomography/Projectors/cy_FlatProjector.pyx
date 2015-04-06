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
		dat_t[:] dat,
		idx_t[:] row,
		idx_t[:] col,
		dat_t[:] vec,
		dat_t[:] res,
		idx_t tensor_length,
		):

	cdef idx_t i

	with nogil:
		for i in range(tensor_length):
			res[col[i]] += dat[i]*vec[row[i]]

def rmatvec(
		dat_t[:] dat,
		idx_t[:] row,
		idx_t[:] col,
		dat_t[:] vec,
		dat_t[:] res,
		idx_t tensor_length,
		):

	cdef idx_t i

	with nogil:
		for i in range(tensor_length):
			res[row[i]] += dat[i]*vec[col[i]]


