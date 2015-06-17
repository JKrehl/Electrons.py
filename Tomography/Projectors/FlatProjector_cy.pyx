#cython: boundscheck=False, initializedcheck=False, wraparound=False

import numpy
cimport numpy
cimport cython
from cython.parallel import parallel, prange
cimport openmp

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
		int threads = 0,
		):

	cdef idx_t tensor_length = dat.size
	cdef idx_t i

	if threads==0:
		threads = openmp.omp_get_max_threads()
	
	with nogil, parallel(num_threads=threads):
		for i in prange(tensor_length, schedule='guided'):
			res[row[i]] += dat[i]*vec[col[i]]


