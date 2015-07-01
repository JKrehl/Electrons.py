#cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True

import numpy
cimport numpy, cython
from cython.parallel import parallel, prange
cimport openmp
from libc.math cimport floor, modf

ctypedef fused itype_t:
	numpy.uint16_t
	numpy.uint32_t
	numpy.uint64_t
	numpy.int16_t
	numpy.int32_t
	numpy.int64_t

ctypedef fused rtype_t:
	numpy.float32_t
	numpy.float64_t

ctypedef fused dtype_t:
	numpy.float32_t
	numpy.float64_t
	numpy.complex64_t

def interpolator2d(itype_t[:] yind, 
				   itype_t[:] xind, 
				   rtype_t[:] yrem, 
				   rtype_t[:] xrem, 
				   dtype_t[:,:] data,
				   int ylength,
				   int xlength,
				   dtype_t fill = 0,
				   int threads=0,
):
    
	if threads==0: threads = openmp.omp_get_num_threads()
	
	cdef int coordlength = yind.size
	
	cdef itype_t yl = ylength, xl = xlength
	
	cdef dtype_t[:] res = numpy.empty(coordlength, numpy.obj2sctype(data))
	
	cdef Py_ssize_t i
	cdef itype_t yi, xi
	cdef rtype_t yr, xr
	cdef dtype_t r
	
	with nogil, parallel(num_threads=threads):
		for i in prange(coordlength, schedule='guided'):
			yi, xi = yind[i], xind[i]
			yr, xr = yrem[i], xrem[i]
			
			if yr<0: 
				yi = yi-1
				yr = 1+yr
			if xr<0: 
				xi = xi-1
				xr = 1+xr
            
			r = 0
            
			if (yi>=0)&(yi<yl):
				if (xi>=0)&(xi<xl):
					r = r + (1-yr)*(1-xr)*data[yi,xi]
				else:
					r = r + (1-yr)*(1-xr)*fill
    
				if (xi+1>=0)&((xi+1)<xl):
					r = r + (1-yr)*xr*data[yi,xi+1]
				else:
					r = r + (1-yr)*xr*fill
                    
			if (yi+1>=0)&((yi+1)<yl):
				if (xi>=0)&(xi<xl):
					r = r + yr*(1-xr)*data[yi+1,xi]
				else:
					r = r + yr*(1-xr)*fill
    
				if (xi+1>=0)&((xi+1)<xl):
					r = r + yr*xr*data[yi+1,xi+1]
				else:
					r = r + yr*xr*fill
            
			res[i] = r

	return numpy.array(res, copy=False)
    

def regularinterpolator2d(itype_t[:] yind, 
				   itype_t[:] xind, 
				   rtype_t[:] yrem, 
				   rtype_t[:] xrem, 
				   dtype_t[:,:] data,
				   int ylength,
				   int xlength,
				   dtype_t fill = 0,
				   int threads=0,
):
    
	if threads==0: threads = openmp.omp_get_num_threads()
	
	cdef int yclength = yind.size
	cdef int xclength = xind.size
	
	cdef itype_t yl = ylength, xl = xlength
	
	cdef dtype_t[:] res = numpy.empty((yclength, xclength), numpy.obj2sctype(data))
	
	cdef Py_ssize_t i
	cdef itype_t yi, xi
	cdef rtype_t yr, xr
	cdef dtype_t r
	
	with nogil, parallel(num_threads=threads):
		for i in prange(yclength, schedule='guided'):
			yi, yr = yind[i], yrem[i]
			
			if yr<0: 
				yi = yi-1
				yr = 1+yr

			if (yi>=0)&(yi<yl):
			
			for j in range(xclength):

				xi, xr = xind[j], xrem[j]
				
				if xr<0: 
					xi = xi-1
					xr = 1+xr
            
			r = 0
            
			if (yi>=0)&(yi<yl):
				if (xi>=0)&(xi<xl):
					r = r + (1-yr)*(1-xr)*data[yi,xi]
				else:
					r = r + (1-yr)*(1-xr)*fill
    
				if (xi+1>=0)&((xi+1)<xl):
					r = r + (1-yr)*xr*data[yi,xi+1]
				else:
					r = r + (1-yr)*xr*fill
                    
			if (yi+1>=0)&((yi+1)<yl):
				if (xi>=0)&(xi<xl):
					r = r + yr*(1-xr)*data[yi+1,xi]
				else:
					r = r + yr*(1-xr)*fill
    
				if (xi+1>=0)&((xi+1)<xl):
					r = r + yr*xr*data[yi+1,xi+1]
				else:
					r = r + yr*xr*fill
            
			res[i] = r

	return numpy.array(res, copy=False)
    
