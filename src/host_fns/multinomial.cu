#include "../header/multinomial.h"
#include "../header/distribution.h"
#include "../header/printing.h"
#include <thrust/scan.h>
#include <thrust/reduce.h>
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


struct exponential{

  __host__ __device__ double operator()(const double &x){
    return exp(x);
  }

};

struct log{
  __host__ __device__ double operator()(const double &x){
    return log(x);
  }
};

void gnl_multinomial(ivec_d &zeta, fvec_d &probs, curandState *states, int K, int G){
  normalize_wts(probs, K, G);
  fvec_d u(G);
  rowIter last_row_iter = getRowIter(K, K-1);
  strideIter strided_iter = thrust::make_permutation_iterator(probs.begin(), last_row_iter);

  thrust::copy(strided_iter, strided_iter + G, u.begin());
  printVec(u, G, 1);
  
  thrust::transform(u.begin(), u.end(), u.begin(), exponential());
  
  printVec(u, G, 1);
  
  double *u_ptr = thrust::raw_pointer_cast(u.data());
  getUniform<<<G, 1>>>(states, u_ptr);

  thrust::transform(u.begin(), u.end(), u.begin(), log());

  gRepEach<realIter>::iterator u_rep = getGRepEachIter(u.begin(), u.end(), K);

  ivec_d dummies(K*G);

  thrust::transform(u_rep, u_rep + K*G, probs.begin(), dummies.begin(), thrust::greater<double>());
  repEachIter colI = getRepEachIter(K, 1);
  thrust::reduce_by_key(colI, colI + K*G, dummies.begin(), thrust::make_discard_iterator(), zeta.begin());
}