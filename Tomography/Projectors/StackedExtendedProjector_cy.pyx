#cython: boundscheck=False, initializedcheck=False, wraparound=False, initializedcheck=False

import numpy
cimport numpy, cython
from cython.parallel import parallel, prange
cimport openmp

ctypedef fused idx_t:
	numpy.int16_t
	numpy.int32_t
	numpy.int64_t
	numpy.uint16_t
	numpy.uint32_t
	numpy.uint64_t

ctypedef fused dat_t:
	numpy.float32_t
	numpy.float64_t
	numpy.complex64_t
	numpy.complex128_t

def matvec(
		dat_t[:] vec,
		dat_t[:] res,
		dat_t[:] dat,
		idx_t[:] idz,
		idx_t[:] bounds,
		idx_t[:] idx_col,
		idx_t[:] idx_row,
		int zlen,
		int threads=0):
	
	cdef idx_t col_stride = numpy.round(vec.size/zlen)
	cdef idx_t row_stride = numpy.round(res.size/zlen)
	cdef idx_t i,j,k

	cdef idx_t idzlen = len(idz)

	cdef dat_t* vec_view
	cdef dat_t* res_view
	cdef dat_t* dat_view
	cdef idx_t* idx_col_view
	cdef idx_t* idx_row_view
	
	if threads==0:
		threads = openmp.omp_get_max_threads()

	with nogil, parallel(num_threads=threads):
		for i in prange(zlen, schedule='guided'):
			res_view = &res[i*row_stride]
			for j in range(idzlen):
				if (i+idz[j])<zlen and (i+idz[j])>=0:
					vec_view = &vec[(i+idz[j])*col_stride]
					dat_view = &dat[bounds[j]]
					idx_col_view = &idx_col[bounds[j]]
					idx_row_view = &idx_row[bounds[j]]
					
					for k in range(bounds[j+1]-bounds[j]):
						res_view[idx_row_view[k]] += dat_view[k]*vec_view[idx_col_view[k]]
