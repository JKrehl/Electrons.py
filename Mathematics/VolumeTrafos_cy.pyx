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

import numpy
cimport numpy, cython
from cython.parallel import parallel, prange
cimport openmp
from libc.math cimport floor, modf

ctypedef fused itype_t:
	numpy.float32_t
	numpy.float64_t
	numpy.float_t

ctypedef fused dtype_t:
	numpy.float32_t
	numpy.float64_t
	numpy.float_t
	numpy.complex64_t

def map_coordinates3(
		dtype_t[:,:,:] source,
		itype_t[:,:,:,:] coords,
		int threads = 0
):

	if threads==0:
		threads = openmp.omp_get_max_threads()

	cdef itype_t[:,:,:] zcoords = coords[0,:,:,:]
	cdef itype_t[:,:,:] ycoords = coords[1,:,:,:]
	cdef itype_t[:,:,:] xcoords = coords[2,:,:,:]
	
	cdef Py_ssize_t szl, syl, sxl
	szl, syl, sxl = source.shape[0:3]

	cdef Py_ssize_t dzl, dyl, dxl
	dzl, dyl, dxl = coords.shape[1:4]
	
	cdef dtype_t[:,:,:] destination = numpy.empty((dzl,dyl,dxl), numpy.array(source, copy=False).dtype)
	cdef Py_ssize_t i,j,k
	
	cdef itype_t sz, sy, sx
	cdef dtype_t d
	cdef Py_ssize_t isz, isy, isx
	cdef itype_t rsz, rsy, rsx
	
	with nogil, parallel(num_threads=threads):
		for i in prange(dzl, schedule='guided'):
			for j in range(dyl):
				for k in range(dxl):
					d = 0
					sz, sy, sx = zcoords[i,j,k], ycoords[i,j,k], xcoords[i,j,k]
					isz, isy, isx = <Py_ssize_t> floor(sz), <Py_ssize_t> floor(sy), <Py_ssize_t> floor(sx)
					rsz, rsy, rsx = sz-floor(sz), sy-floor(sy), sx-floor(sx)
					
					if isz>=0 and isy>=0 and isx>=0 and isz<szl-1 and isy<syl-1 and isx<sxl-1:
						d = (1-rsz)*(1-rsy)*(1-rsx)*source[isz  ,isy  ,isx  ] +\
							(1-rsz)*(1-rsy)*(  rsx)*source[isz  ,isy  ,isx+1] +\
							(1-rsz)*(  rsy)*(1-rsx)*source[isz  ,isy+1,isx  ] +\
							(1-rsz)*(  rsy)*(  rsx)*source[isz  ,isy+1,isx+1] +\
							(  rsz)*(1-rsy)*(1-rsx)*source[isz+1,isy  ,isx  ] +\
							(  rsz)*(1-rsy)*(  rsx)*source[isz+1,isy  ,isx+1] +\
							(  rsz)*(  rsy)*(1-rsx)*source[isz+1,isy+1,isx  ] +\
							(  rsz)*(  rsy)*(  rsx)*source[isz+1,isy+1,isx+1]
						
					destination[i,j,k] = d
					
	return numpy.array(destination, copy=False)
