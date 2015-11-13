#cython: boundscheck=False, initializedcheck=False, wraparound=False

cimport numpy
cimport cython
from cython.parallel cimport parallel, prange
cimport openmp

numpy.import_array()

from Projector_Utilities cimport idx_t, dat_t

def matvec(
		dat_t[:] vec,
		dat_t[:] res,
		dat_t[:] dat,
		idx_t[:] bounds,
		idx_t[:] idx,
		int threads = 0,
		):

	cdef numpy.npy_intp length = bounds.size-1 
	cdef numpy.npy_intp i,j
	cdef dat_t tmp

	if threads==0:
		threads = openmp.omp_get_max_threads()
	
	with nogil, parallel(num_threads=threads):
		for i in prange(length, schedule='guided'):
			tmp = 0
			for j in range(bounds[i], bounds[i+1]):
				tmp = tmp+dat[j]*vec[idx[j]]
			res[i] = tmp
