/*

Copyright (c) 2015 Jonas Krehl <Jonas.Krehl@triebenberg.de>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

#include <omp.h>
#include <stdio.h>

#include <complex>

#include <boost/preprocessor/repetition/repeat.hpp>

#include "atomic_add.hpp"

template <typename DTYPE, typename ITYPE>
void sparse_matvec_template(PyArrayObject* inp, PyArrayObject* outp, npy_intp zsize, PyArrayObject* dz, PyArrayObject* bounds, PyArrayObject* col, PyArrayObject* row, PyArrayObject* coeff, int threads)
{

	PyArrayObject* ops[3] = {col, row, coeff};
	npy_uint32 flags = NPY_ITER_EXTERNAL_LOOP|NPY_ITER_GROWINNER|NPY_ITER_RANGED|NPY_ITER_BUFFERED|NPY_ITER_DELAY_BUFALLOC;
	npy_uint32 op_flags[3] = {NPY_ITER_READONLY, NPY_ITER_READONLY, NPY_ITER_READONLY};
	NpyIter* iter = NpyIter_MultiNew(3, ops , flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags, NULL);

	npy_intp dz_size = PyArray_SIZE(dz);
	npy_intp inp_zstride = PyArray_SIZE(inp)/zsize;
	npy_intp outp_zstride = PyArray_SIZE(outp)/zsize;

	ITYPE* dz_da = (ITYPE*) dz->data;
	ITYPE* bounds_da = (ITYPE*) bounds->data;
	DTYPE* inp_da = (DTYPE*) inp->data;
	DTYPE* outp_da = (DTYPE*) outp->data;
	DTYPE* inp_loc;
	DTYPE* outp_loc;

	NpyIter* local_iter;
	NpyIter_IterNextFunc* iternext;

	npy_intp loopsize;
	ITYPE* loop_col;
	ITYPE* loop_row;
	DTYPE* loop_coeff;
	npy_intp loop_col_stride;
	npy_intp loop_row_stride;
	npy_intp loop_coeff_stride;

	npy_intp ib, ie, iz, idz, oz;
	int i;

	#pragma omp parallel num_threads(threads) private(local_iter, iternext)
	{

		#pragma omp critical
		local_iter = NpyIter_Copy(iter);

		iternext = NpyIter_GetIterNext(local_iter, NULL);

		#pragma omp for collapse(2) private(iz, idz, oz, inp_loc, outp_loc, ib, ie, loopsize, loop_col, loop_row, loop_coeff, loop_col_stride, loop_row_stride, loop_coeff_stride, i)
		for(iz=0; iz<zsize; iz++)
		{
			for(idz=0; idz<dz_size; idz++)
			{
				oz = iz + dz_da[idz];
				if(oz>=0 && oz<zsize)
				{
					inp_loc = inp_da + iz*inp_zstride;
					outp_loc = outp_da + oz*outp_zstride;
					ib = bounds_da[idz];
					ie = bounds_da[idz+1];

					if(ie>ib)
					{
						NpyIter_ResetToIterIndexRange(local_iter, ib, ie, NULL);

						do{
							loopsize = NpyIter_GetInnerLoopSizePtr(local_iter)[0];

							loop_col = (ITYPE*) NpyIter_GetDataPtrArray(local_iter)[0];
							loop_row = (ITYPE*) NpyIter_GetDataPtrArray(local_iter)[1];
							loop_coeff = (DTYPE*) NpyIter_GetDataPtrArray(local_iter)[2];

							loop_col_stride = NpyIter_GetInnerStrideArray(local_iter)[0]/sizeof(ITYPE);
							loop_row_stride = NpyIter_GetInnerStrideArray(local_iter)[1]/sizeof(ITYPE);
							loop_coeff_stride = NpyIter_GetInnerStrideArray(local_iter)[2]/sizeof(DTYPE);

							#pragma omp simd
							for(i=0; i<loopsize; i++)
							{
								atomic_add(outp_loc + loop_row[0], loop_coeff[0] * inp_loc[loop_col[0]]);

								loop_col += loop_col_stride;
								loop_row += loop_row_stride;
								loop_coeff += loop_coeff_stride;
							}

						} while (iternext(local_iter));
					}
				}
			}
		}

		NpyIter_Deallocate(local_iter);

	}

	NpyIter_Deallocate(iter);
}

#define DTYPES(I) DTYPES ## I
#define DTYPES0 npy_float32
#define DTYPES1 npy_float64
#define DTYPES2 std::complex<npy_float32>
#define DTYPES3 std::complex<npy_float64>

#define DTYPENAMES(I) DTYPENAMES ## I
#define DTYPENAMES0 NPY_FLOAT32
#define DTYPENAMES1 NPY_FLOAT64
#define DTYPENAMES2 NPY_COMPLEX64
#define DTYPENAMES3 NPY_COMPLEX128

#define DTYPES_CNT 4

#define ITYPES(I) ITYPES ## I
#define ITYPES0 npy_int16
#define ITYPES1 npy_int32
#define ITYPES2 npy_int64
#define ITYPES3 npy_uint16
#define ITYPES4 npy_uint32
#define ITYPES5 npy_uint64

#define ITYPENAMES(I) ITYPENAMES ## I
#define ITYPENAMES0 NPY_INT16
#define ITYPENAMES1 NPY_INT32
#define ITYPENAMES2 NPY_INT64
#define ITYPENAMES3 NPY_UINT16
#define ITYPENAMES4 NPY_UINT32
#define ITYPENAMES5 NPY_UINT64

#define ITYPES_CNT 6

#define DISPATCH_INNER_CASE(z, itype, dtype)							\
	case ITYPENAMES(itype): return sparse_matvec_template<DTYPES(dtype), ITYPES(itype)>;

#define DISPATCH_CASE(z, dtype, itype_num)								\
	case DTYPENAMES(dtype): switch(itype_num) { BOOST_PP_REPEAT(ITYPES_CNT, DISPATCH_INNER_CASE, dtype) };

void (*dispatch_sparse_matvec(int dtype_num, int itype_num))(PyArrayObject*, PyArrayObject*, npy_intp, PyArrayObject*, PyArrayObject*, PyArrayObject*, PyArrayObject*, PyArrayObject*, int)
{
	switch(dtype_num){ BOOST_PP_REPEAT(DTYPES_CNT, DISPATCH_CASE, itype_num) };
}

#undef DISPATCH_INNER_CASE
#undef DISPATCH_CASE

void sparse_matvec(PyArrayObject* inp, PyArrayObject* outp, npy_intp zsize, PyArrayObject* dz, PyArrayObject* bounds, PyArrayObject* col, PyArrayObject* row, PyArrayObject* coeff, int threads)
{
	dispatch_sparse_matvec(inp->descr->type_num, col->descr->type_num)(inp, outp, zsize, dz, bounds, col, row, coeff, threads);
}
