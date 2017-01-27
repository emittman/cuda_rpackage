#ifndef SUMMARY2_H
#define SUMMARY2_H

#include "iterator.h"
#include "chain.h"
#include "printing.h"
#include "distribution.h"
#include "curand_kernel.h"
#include "transpose.h"
#include "beta_hat.h"
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>

struct mult_scalar_by_sqrt{
  template <typename T>
  __host__ __device__ void operator()(T tup){
    thrust::get<0>(tup) *= 1 / sqrt(thrust::get<1>(tup));
  }
};

struct add3{
  template <typename T>
  __host__ __device__ void operator()(T tup){
  thrust::get<0>(tup) = thrust::get<0>(tup) + thrust::get<1>(tup) + thrust::get<2>(tup);
  }
};

struct summary2{
  
  int G;
  int K;
  int V;
  ivec_d occupied;
  ivec_d unoccupied;
  int num_occupied;
  ivec_d Mk;
  fvec_d yty_sums;
  fvec_d xty_sums;
  fvec_d ytx_sums;
  
  summary2(int _K, ivec_d zeta, data_t &dat);
  void print_Mk(){ printVec(Mk, 1, K);}
  void print_yty(){ printVec(yty_sums, 1, num_occupied);}
  void print_xty(){ printVec(xty_sums, V, num_occupied);}
  void draw_MVNormal(curandState *states, fvec_d &beta_hat, fvec_d &chol_prec, fvec_d &beta, priors_t &priors);
  void sumSqErr(fvec_d &sse, fvec_d &beta, fvec &xtx);
};

#endif