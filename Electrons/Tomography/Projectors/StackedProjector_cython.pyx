#cython: boundscheck=False, initializedcheck=False, wraparound=False
"""
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
"""

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
