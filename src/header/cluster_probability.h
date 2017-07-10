#ifndef CLUST_PROB
#define CLUST_PROB

#include "chain.h"
#include "iter_getter.h"
#include "quadform2.h"

// used here to compute all combinations of t(beta_k) %*% t(X) %*% y_g
void big_matrix_multiply(fvec_d &A, fvec_d &B, fvec_d &big_grid, int a1, int a2, int b1, int b2);

// thrust boilerplate to use for_each on zip_iterator
typedef thrust::tuple<fvec_d::iterator, gRepTimes<realIter>::iterator,
                      gRepTimes<realIter>::iterator, gRepEach<realIter>::iterator,
                      gRepTimes<realIter>::iterator> weight_tup_voom;
typedef thrust::tuple<fvec_d::iterator, gRepTimes<realIter>::iterator,
                      gRepTimes<realIter>::iterator, gRepEach<realIter>::iterator,
                      gRepTimes<realIter>::iterator> weight_tup_no_voom;

typedef thrust::zip_iterator<weight_tup_voom> weight_zip_voom;
typedef thrust::zip_iterator<weight_tup_no_voom> weight_zip_no_voom;

typedef thrust::tuple<double &, double &, double &, double&, double&> weight_tup_el;

struct clust_prob{
  int n;
  clust_prob(int _n): n(_n){};
  __host__ __device__ void operator()(weight_tup_el Tup);
};  

void cluster_weights_voom(fvec_d &big_grid, data_t &data, chain_t &chain, int verbose);
void cluster_weights_no_voom(fvec_d &big_grid, data_t &data, chain_t &chain, int verbose);


#endif