#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "iter_getter.h"

template<typename T>
void transpose(T in_begin, T in_end, int R, int C, T out_begin){
  typename gTranspose<T>::iterator iter = getGTransposeIter(in_begin, in_end, R, C);
  thrust::copy(iter, iter+R*C, out_begin);
}

#endif
