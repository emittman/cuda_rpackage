#ifndef CONSTRUCT_PREC_H
#define CONSTRUCT_PREC_H

#include "iterator.h"
#include "iter_getter.h"

typedef thrust::tuple<gDiagonal<realIter>::iterator,gRepTimes<realIter>::iterator,gRepEach<realIter>::iterator> diag_tup;
typedef thrust::zip_iterator<diag_tup> diag_zip;
typedef thrust::tuple<double &, double &, double &> diag_tup_el;

struct diagAdd{
  __host__ __device__ void operator()(diag_tup_el Tup);
};  

void construct_prec(realIter prec_begin, realIter prec_end, realIter lam_begin, realIter lam_end,
                    realIter tau_begin, realIter tau_end, intIter Mk_begin, intIter Mk_end,
                    realIter xtx_begin, realIter xtx_end, int K, int V);


#endif
