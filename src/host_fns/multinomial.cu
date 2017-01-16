#include "../header/multinomial.h"
#include <thrust/scan.h>
#include <thrust/functional.h>


__host__ __device__ double log_sum_exp::operator()(double &x, double &y){
  double M = max(x, y);
  return log(exp(y-M) + exp(x-M)) + M;
}

void normalize_wts(fvec_d &big_grid, int K, int G){
  repEachIter key = getRepEachIter(K, 1);
  log_sum_exp f;
  thrust::inclusive_scan_by_key(key, key + K*G, big_grid.begin(), big_grid.begin(), thrust::equal_to<int>(), f);
}