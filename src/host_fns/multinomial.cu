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


void gnl_multinomial(ivec_d &zeta, fvec_d &probs, curandState *states, int K, int G){
  normalize_wts(probs, K, G);
  fvec_d u(G);
  rowIter last_row_iter = getRowIter(K, K-1);
  strideIter strided_iter = thrust::make_permutation_iterator(probs.begin(), last_row_iter);

  thrust::copy(strided_iter, strided_iter + G, u.begin());
  //printVec(u, G, 1);
  
  double *u_ptr = thrust::raw_pointer_cast(u.data());
  getUniform<<<G, 1>>>(states, G, u_ptr);
  //std::cout << "u:sampled:\n";
  //printVec(u, G, 1);
  //std::cout << "probs...:\n";
  //printVec(probs, K, G);
  gRepEach<realIter>::iterator u_rep = getGRepEachIter(u.begin(), u.end(), K);

  ivec_d dummies(K*G);
  thrust::transform(u_rep, u_rep + K*G, probs.begin(), dummies.begin(), thrust::greater<double>());
  //std::cout << "dummies:\n";
  //printVec(dummies, K, G);
  repEachIter colI = getRepEachIter(K, 1);
  thrust::reduce_by_key(colI, colI + K*G, dummies.begin(), thrust::make_discard_iterator(), zeta.begin());
}