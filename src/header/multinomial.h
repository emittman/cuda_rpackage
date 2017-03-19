#ifndef MULTINOM
#define MULTINOM

#include "curand_kernel.h"
#include "iter_getter.h"

struct exponential{
  
  __host__ __device__ double operator()(const double &x){
    return exp(x);
  }
  
};

struct logorithmic{
  __host__ __device__ double operator()(const double &x){
    return log(x);
  }
};

struct log_sum_exp{
  __host__ __device__ double operator()(double &x, double &y){
    double M = max(x, y);
    return log(exp(y-M) + exp(x-M)) + M;
  }
};

void normalize_wts(fvec_d &big_grid, int K, int G);

void gnl_multinomial(ivec_d &zeta, fvec_d &probs, curandState *states, int K, int G);

#endif
