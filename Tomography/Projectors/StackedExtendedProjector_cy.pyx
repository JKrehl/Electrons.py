#cython: boundscheck=False, initializedcheck=False, wraparound=False, initializedcheck=False

import numpy
cimport numpy, cython
from cython.parallel import parallel, prange
cimport openmp

numpy.import_array()

from Projector_Utilities cimport idx_t, dat_t, atomic_add

cdef extern from "StackedExtendedProjector_cpp.cpp" nogil:
	void sparse_matvec(numpy.ndarray inp, numpy.ndarray outp, numpy.npy_intp zsize, numpy.ndarray dz, numpy.ndarray bounds, numpy.ndarray col, numpy.ndarray row, numpy.ndarray coeff, int threads)

DTYPES = (numpy.float32, numpy.float64, numpy.complex64, numpy.complex128)
ITYPES = (numpy.int16, numpy.int32, numpy.int64, numpy.uint16, numpy.uint32, numpy.uint64)
	
def matvec(
		numpy.ndarray vec,
		numpy.ndarray res,
		numpy.npy_intp zsize,
		numpy.ndarray dz,
		numpy.ndarray bounds,
		numpy.ndarray col,
		numpy.ndarray row,
		numpy.ndarray coeff,
		int threads=0):
	
	assert res.dtype==vec.dtype and coeff.dtype==vec.dtype, "all data-containing arrays should have the same Datatype: vec: %s, res: %s, coeff: %s"%(res.dtype, vec.dtype, coeff.dtype)
	assert res.dtype in DTYPES, "type of data-arrays not acceptable: %s"%res.dtype

	assert dz.dtype==bounds.dtype and dz.dtype==col.dtype and dz.dtype==row.dtype, "all indices-containing arrays should have the same Datatype: idz: %s, bounds: %s, col: %s, row: %s"%(dz.dtype, bounds.dtype, col.dtype, row.dtype)
	assert dz.dtype in ITYPES, "type of indices-arrays not acceptable: %s"%dz.dtype

	if threads==0:
		threads = openmp.omp_get_max_threads()

	sparse_matvec(vec, res, zsize, dz, bounds, col, row, coeff, threads)

def matvec2(
		dat_t[:] vec,
		dat_t[:] res,
		numpy.npy_intp zsize,
		idx_t[:] idz,
		idx_t[:] bounds,
		idx_t[:] idx_col,
		idx_t[:] idx_row,
		dat_t[:] coeff,
		int threads=0):
	
	cdef idx_t col_stride = numpy.round(vec.size/zsize)
	cdef idx_t row_stride = numpy.round(res.size/zsize)
	cdef idx_t i,j,k

	cdef idx_t idzlen = len(idz)

	cdef dat_t* vec_view
	cdef dat_t* res_view
	cdef dat_t* coeff_view
	cdef idx_t* idx_col_view
	cdef idx_t* idx_row_view
	
	if threads==0:
		threads = openmp.omp_get_max_threads()

	with nogil, parallel(num_threads=threads):
		for i in prange(zsize, schedule='guided'):
			res_view = &res[i*row_stride]
			for j in range(idzlen):
				if (i+idz[j])<zsize and (i+idz[j])>=0 and bounds[j+1]>bounds[j]:
					vec_view = &vec[(i+idz[j])*col_stride]
					coeff_view = &coeff[bounds[j]]
					idx_col_view = &idx_col[bounds[j]]
					idx_row_view = &idx_row[bounds[j]]
					
					for k in range(bounds[j+1]-bounds[j]):
						atomic_add(&res_view[idx_row_view[k]], coeff_view[k]*vec_view[idx_col_view[k]])
