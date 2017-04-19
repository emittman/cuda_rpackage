#ifndef CONSTRUCT_PREC_H
#define CONSTRUCT_PREC_H

#include "iterator.h"
#include "iter_getter.h"
#include "chain.h"
#include "summary2.h"
#include <thrust/for_each.h>

typedef thrust::tuple<gDiagonal<realIter>::iterator,gRepTimes<realIter>::iterator> diag_tup;
typedef thrust::zip_iterator<diag_tup> diag_zip;
typedef thrust::tuple<double &, double &> diag_tup_el;

struct diagAdd{
  __host__ __device__ void operator()(diag_tup_el Tup);
};  

void construct_prec(fvec_d &prec, summary2 &smry, priors_t &priors, chain_t &chain, int verbose);

typedef thrust::tuple<realIter, gRepEach<realIter>::iterator, gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator> wt_sum_tup;
typedef thrust::zip_iterator<wt_sum_tup> wt_sum_zip;
typedef thrust::tuple<double &, double &, double &, double &> wt_sum_el;

struct weighted_sum_functor{
  __host__ __device__ void operator()(wt_sum_el tup);
};

void construct_weighted_sum(fvec_d &weighted_sum, summary2 &smry, priors_t &priors, chain_t &chain, int verbose);

#endif
