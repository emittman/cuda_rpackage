#include "../header/multinomial.h"
#include "../header/distribution.h"
#include "../header/printing.h"
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

void normalize_wts(fvec_d &big_grid, int K, int G){
  repEachIter key = getRepEachIter(K, 1);
  log_sum_exp f;
  thrust::inclusive_scan_by_key(key, key + K*G, big_grid.begin(), big_grid.begin(), thrust::equal_to<int>(), f);
}

struct compare_eval{
  template <typename T>
  __host__ __device__ void operator()(T tup){
    thrust::get<0>(tup) = thrust::get<0>(tup) < thrust::get<1>(tup) ? 1 : 0;
  }
};

void gnl_multinomial(ivec_d &zeta, fvec_d &probs, curandState *states, int K, int G){
  normalize_wts(probs, K, G);
  
  /*copy last value in cumulative sums to a vector to be used to scale U(0,1) draws*/
  fvec_d u(G);
  rowIter last_row_iter = getRowIter(K, K-1);
  strideIter strided_iter = thrust::make_permutation_iterator(probs.begin(), last_row_iter);
  thrust::copy(strided_iter, strided_iter + G, u.begin());
  //printVec(u, G, 1);
  
  /*draw uniforms*/
  double *u_ptr = thrust::raw_pointer_cast(u.data());
  getUniform<<<G, 1>>>(states, G, u_ptr);
  //std::cout << "u:sampled:\n";
  //printVec(u, G, 1);
  //std::cout << "probs...:\n";
  //printVec(probs, K, G);
  gRepEach<realIter>::iterator u_rep = getGRepEachIter(u.begin(), u.end(), K);

  
  //ivec_d dummies(K*G);
  typedef thrust::tuple<realIter, gRepEach<realIter>::iterator> mult_tup;
  typedef thrust::zip_iterator<mult_tup> mult_zip;
  mult_tup mytup = thrust::tuple<realIter, gRepEach<realIter>::iterator>(probs.begin(), u_rep);
  mult_zip myzip = thrust::zip_iterator<mult_tup>(mytup);
  compare_eval f;
  thrust::for_each(myzip, myzip + K*G, f);
  //thrust::transform(u_rep, u_rep + K*G, probs.begin(), dummies.begin(), thrust::greater<double>());
  //std::cout << "dummies:\n";
  //printVec(dummies, K, G);
  repEachIter colI = getRepEachIter(K, 1);
  thrust::reduce_by_key(colI, colI + K*G, probs.begin(), thrust::make_discard_iterator(), zeta.begin());
}