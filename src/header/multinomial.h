#ifndef MULTINOM
#define MULTINOM

#include "curand_kernel.h"
#include "iter_getter.h"


struct log_sum_exp{
  __host__ __device__ double operator()(double &x, double &y);
};

void normalize_wts(fvec_d &big_grid, int K, int G);

void gnl_multinomial(ivec_d &zeta, fvec_d &probs, curandState *states, int K, int G);

#endif
