#include <omp.h> 
#include <complex>

#ifndef ATOMIC_ADD
#define ATOMIC_ADD

template <typename type>
void atomic_add_specialised( type* a, type b )
{
#pragma omp atomic update
	a[0] += b;
}

template <typename T>
inline void atomic_add_specialised(std::complex<T>* dest, std::complex<T> val)
{
  T* split_dest = reinterpret_cast<T(&)[2]>( dest[0] );
	
  #pragma omp atomic update
  split_dest[0] += val.real();
  #pragma omp atomic update
  split_dest[1] += val.imag();
}

template <typename T>
inline void atomic_add(T* dest, T val)
{
  atomic_add_specialised(dest, val);
}

#endif
