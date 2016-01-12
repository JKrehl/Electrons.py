cimport numpy

ctypedef fused itype:
	numpy.int16_t
	numpy.int32_t
	numpy.int64_t
	numpy.uint16_t
	numpy.uint32_t
	numpy.uint64_t

ctypedef fused dtype:
	numpy.float32_t
	numpy.float64_t
	numpy.complex64_t
	numpy.complex128_t
	
cdef extern from "source/atomic_add.hpp" nogil:
	inline void atomic_add[T](T*, T)
