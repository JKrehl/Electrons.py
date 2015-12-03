#cython: boundscheck=False, initializedcheck=False, wraparound=False

cimport numpy
cimport cython
from cython.parallel cimport parallel, prange
cimport openmp

numpy.import_array()

from Projector_Utilities cimport itype, dtype

def matvec(
		dtype[:] vec,
		dtype[:] res,
		dtype[:] dat,
		itype[:] bounds,
		itype[:] idx,
		int threads = 0,
		):

	cdef numpy.npy_intp length = bounds.size-1 
	cdef numpy.npy_intp i,j
	cdef dtype tmp

	if threads==0:
		threads = openmp.omp_get_max_threads()
	
	with nogil, parallel(num_threads=threads):
		for i in prange(length, schedule='guided'):
			tmp = 0
			for j in range(bounds[i], bounds[i+1]):
				tmp = tmp+dat[j]*vec[idx[j]]
			res[i] = tmp
