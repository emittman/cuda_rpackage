#ifndef CONSTRUCT_PREC_H
#define CONSTRUCT_PREC_H

#include "header/iterator.h"

typedef thrust::tuple<gDiagonal<realIter>::iterator,gRepTimes<realIter>::iterator,gRepEach<realIter>::iterator> diag_tup;
typedef thrust::zip_iterator<diag_tup> diag_zip;
typedef thrust::tuple<double &, double &, double &> diag_tup_el;

struct diagAdd{
  __host__ __device__ void operator()(diag_tup_el Tup){
    thrust::get<0>(Tup) = thrust::get<0>(Tup) + thrust::get<1>(Tup)/thrust::get<2>(Tup);
  }
};  

void construct_prec(realIter prec_begin, realIter prec_end, realIter lam_begin, realIter lam_end, realIter tau_begin, realIter tau_end, intIter Mk_begin, intIter Mk_end, realIter xtx_begin, realIter xtx_end, int K, int V){

  if(thrust::distance(prec_begin, prec_end) < K*V*V) std::cout <<"DIMENSION MISMATCH!\n";

  //copy xtx over in a repeating loop (initialization)
  gRepTimes<realIter>::iterator xtx_rep = getGRepTimesIter(xtx_begin, xtx_end, V*V, 1); 
  thrust::copy(xtx_rep, xtx_rep + K*V*V, prec_begin);
  
  //multiply by Mk
  gRepEach<intIter>::iterator Mk_rep = getGRepEachIter(Mk_begin, Mk_end, V*V, 1);
  transform(prec_begin, prec_end, Mk_rep, prec_begin, thrust::multiplies<double>());

  //modify diagonal according to prior prec/error prec
  diagAdd f;
  gDiagonal<realIter>::iterator prec_diag = getGDiagIter(prec_begin, prec_end, V);
  gRepTimes<realIter>::iterator lam_iter  = getGRepTimesIter(lam_begin, lam_end, V, 1);
  gRepEach<realIter>::iterator   tau_iter  = getGRepEachIter(tau_begin, tau_end, V, 1);
  diag_zip zipped = thrust::zip_iterator<diag_tup>(thrust::make_tuple(prec_diag, lam_iter, tau_iter));
  thrust::for_each(zipped, zipped+K*V, f);
  
  /*
  */

}

#endif
