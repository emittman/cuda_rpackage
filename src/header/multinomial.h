#ifndef MULTINOM
#define MULTINOM

#include "iter_getter.h"

struct log_sum_exp{
  __host__ __device__ double operator()(double &x, double &y);
};

void normalize_wts(fvec_d &big_grid, int K, int G);

#endif
