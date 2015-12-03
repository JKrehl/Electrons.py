#cython: boundscheck=False, initializedcheck=False, wraparound=False

cimport numpy
cimport cython
from cython.parallel cimport parallel, prange
cimport openmp

numpy.import_array()

from Projector_Utilities cimport itype, dtype, atomic_add

def matvec(
		dtype[:] vec,
		dtype[:] res,
		dtype[:] dat,
		itype[:] row,
		itype[:] col,
		int threads = 0,
		):

	cdef numpy.npy_intp tensor_length = dat.size
	cdef numpy.npy_intp i
	cdef dtype tmp

	if threads==0:
		threads = openmp.omp_get_max_threads()
	
	with nogil, parallel(num_threads=threads):
		for i in prange(tensor_length, schedule='guided'):
			tmp = dat[i]*vec[row[i]]
			atomic_add(&res[col[i]], tmp)

