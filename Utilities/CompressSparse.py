import numpy

def compress_sparse(vec, size, *arrays):
	csvec = numpy.empty(size+1, vec.dtype)
	cs_arrays = tuple(numpy.empty(a.shape, a.dtype) for a in arrays)
	
	csvec[0] = 0

	for i in range(size):
		s = numpy.where(vec==i)[0]
		lb, ub = csvec[i], csvec[i]+s.size
		for a, csa in zip(arrays, cs_array):
			csa[lb:ub] = a[s]

		csvec[i+1] = ub

	return (csvec,)+cs_arrays
