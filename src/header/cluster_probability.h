#ifndef CLUST_PROB
#define CLUST_PROB

#include "iter_getter.h"

typedef thrust::tuple<fvec_d::iterator,gRepTimes<realIter>::iterator,gRepTimes<realIter>::iterator, gRepEach<realIter>::iterator, gRepEach<realIter>::iterator> weight_tup;
typedef thrust::zip_iterator<weight_tup> weight_zip;
typedef thrust::tuple<double &, double &, double &, double&, double&> weight_tup_el;

struct clust_prob{
  int n;
  clust_prob(int _n): n(_n){};
  __host__ __device__ void operator()(weight_tup_el Tup);
};  

void big_matrix_multiply(fvec_d &A, fvec_d &B, fvec_d &big_grid, int a1, int a2, int b1, int b2);

void cluster_weights(fvec &big_grid, fvec &pi, fvec &tau2, fvec &yty, fvec &bxxb);


#endif