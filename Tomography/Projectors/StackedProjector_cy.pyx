#cython: boundscheck=False, initializedcheck=False, wraparound=False

import numpy
cimport numpy, cython
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
		idx_t stack_height,
		idx_t row_stride,
		idx_t col_stride,
		int threads=0):
	
	cdef idx_t tensor_length = dat.size
	cdef idx_t i,j

	cdef dat_t* vec_view
	cdef dat_t* res_view
	
	if threads==0:
		threads = openmp.omp_get_max_threads()

	with nogil, parallel(num_threads=threads):
		for i in prange(stack_height, schedule='dynamic'):
			vec_view = &vec[i*col_stride]
			res_view = &res[i*row_stride]

			for j in range(tensor_length):
				res_view[row[j]] += dat[j]*vec_view[col[j]]
