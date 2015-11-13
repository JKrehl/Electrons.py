cimport numpy

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
	
cdef extern from "atomic_add.hpp" nogil:
	inline void atomic_add[T](T*, T)
