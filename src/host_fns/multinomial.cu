#include "../header/multinomial.h"
#include "../header/distribution.h"
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

__host__ __device__ void is_greater::operator()(compare_tup_el Tup){
  thrust::get<2>(Tup) = log(thrust::get<0>(Tup)) > thrust::get<1> ? 1 : 0;
}

typedef thrust::permutation_iterator<fvec_d::iterator, repTimesIter> strideIter;

void gnl_multinomial(ivec_d &zeta, fvec_d &probs, curandState *states, int K, int G){
  normalize_wts(probs, K, G);
  fvec_d u(G);
  repTimesIter last_row_iter = getRepTimesIter(G, K);
  strideIter strided_iter = thrust::make_permutation_iterator(probs.begin(), las_row_iter);
  thrust::copy(thrust::make_transform_iterator(strided_iter, exp<double>()),
               thrust::make_transform_iterator(strided_iter, exp<double>()) + G,
               u.begin());
               
  double *u_ptr = thrust::raw_pointer_cast(u.data());
  getUniform<<<n, 1>>>(states, u_ptr);
  ivec_d dummies(K*G);
  gRepEach<realIter>::iterator u_rep = getGRepEachIter(u.begin(), u.end(), K);
  compare_zip zipped = thrust::zip_iterator<compare_tup>(thrust::make_tuple(u_rep, probs.begin(), dummies.begin()));
  is_greater f;
  thrust::for_each(zipped, zipped + K*G, f);
  
  repEachIter colI = getRepEachIter(K, 1);
  thrust::reduce_by_key(colI, colI + K*G, dummies.begin(), zeta.begin());
}