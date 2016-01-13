#cython: boundscheck=False, initializedcheck=False, wraparound=False, initializedcheck=False
#distutils: include_dirs = [numpy.get_include(), /home/krehl/Python/Electrons/Tomography/Projectors/]
#distutils: extra_compile_args = [-fopenmp, -O3, -march=native]
#distutils: extra_link_args = [-fopenmp,]
#distutils: language = c++

cimport numpy
from ..Tomography.Projectors.Projector_Utilities import itype, dtype

def CS_pointers(numpy.ndarray[itype, ndim=1] vec, numpy.npy_intp size):
	cdef numpy.ndarray[itype, ndim=1] pointers = numpy.zeros(size+1, vec.dtype)

	pointers[0] = 0

	cdef numpy.npy_intp i, pi

	for i in range(vec.size):
		pointers[vec[i]+1] += 1

	pi = 0
	for i in range(size+1):
		pi += pointers[i]
		pointers[i] = pi

	return pointers