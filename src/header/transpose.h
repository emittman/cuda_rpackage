#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "iter_getter.h"

template<typename T1, typename T2>
void transpose(T1 in_begin, T1 in_end, int R, int C, T2 out_begin){
  typename gTranspose<T1>::iterator iter = getGTransposeIter(in_begin, in_end, R, C);
  thrust::copy(iter, iter+R*C, out_begin);
}

#endif
