#ifndef MULTINOM
#define MULTINOM

#include "iter_getter.h"
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


struct log_sum_exp{
  __host__ __device__ double operator()(double &x, double &y){
    double M = max(x, y);
    return log(exp(y-M) + exp(x-M)) + M;
  }
};

void normalize_wts(fvec_d &big_grid, int K, int G){
  repEachIter key = getRepEachIter(K, 1);
  log_sum_exp f;
  thrust::inclusive_scan_by_key(key, key + K*G, big_grid.begin(), big_grid.begin(), thrust::equal_to<int>(), f);
}

#endif
