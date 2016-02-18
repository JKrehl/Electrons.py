#cython: boundscheck=False, initializedcheck=False, wraparound=False

from cython.parallel import parallel, prange
cimport openmp

from ...Utilities.Projector_Utilities cimport itype, dtype

def matvec(
		dtype[:] vec,
		dtype[:] res,
		dtype[:] dat,
		itype[:] col,
		itype[:] row,
		itype stack_height,
		itype col_stride,
		itype row_stride,
		int threads=0):
	
	cdef itype tensor_length = dat.size
	cdef itype i,j

	cdef dtype* vec_view
	cdef dtype* res_view
	
	if threads==0:
		threads = openmp.omp_get_max_threads()

	with nogil, parallel(num_threads=threads):
		for i in prange(stack_height, schedule='guided'):
			vec_view = &vec[i*col_stride]
			res_view = &res[i*row_stride]

			for j in range(tensor_length):
				res_view[row[j]] += dat[j]*vec_view[col[j]]
