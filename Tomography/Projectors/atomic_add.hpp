#include <omp.h> 

template <typename type>
void atomic_add( type* a, type b )
{
#pragma omp atomic update
	a[0] += b;
}