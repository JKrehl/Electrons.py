cimport numpy
cimport numpy

cpdef numpy.npy_intp where_first(numpy.ndarray[numpy.uint8_t, cast=True, ndim=1] arr):
	cdef numpy.npy_intp i = 0
	for i in range(arr.size):
		if arr[i]==1: return i
	return -1

cpdef numpy.npy_intp where_first_not(numpy.ndarray[numpy.uint8_t, cast=True, ndim=1] arr):
	cdef numpy.npy_intp i = 0
	for i in range(arr.size):
		if arr[i]==0: return i
	return -1

cpdef numpy.npy_intp where_last(numpy.ndarray[numpy.uint8_t, cast=True, ndim=1] arr):
	cdef numpy.npy_intp i = 0
	for i in range(arr.size-1, -1, -1):
		if arr[i]==1: return i
	return -1

cpdef numpy.npy_intp where_last_not(numpy.ndarray[numpy.uint8_t, cast=True, ndim=1] arr):
	cdef numpy.npy_intp i = 0
	for i in range(arr.size-1, -1, -1):
		if arr[i]==0: return i
	return -1
