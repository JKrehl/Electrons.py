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
	numpy.int16_t
	numpy.int32_t
	numpy.int64_t

def matvec(
		dat_t[:] vec,
		dat_t[:] res,
		dat_t[:] dat,
		idx_t[:] bounds,
		idx_t[:] idx,
		int threads = 0,
		):

	cdef idx_t length = bounds.size-1 
	cdef idx_t i,j
	cdef dat_t tmp

	if threads==0:
		threads = openmp.omp_get_max_threads()
	
	with nogil, parallel(num_threads=threads):
		for i in prange(length, schedule='guided'):
			tmp = 0
			for j in range(bounds[i], bounds[i+1]):
				tmp = tmp+dat[j]*vec[idx[j]]
			res[i] = tmp
