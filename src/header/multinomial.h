#ifndef MULTINOM
#define MULTINOM

#include "curand_kernel.h"
#include "iter_getter.h"


struct log_sum_exp{
  __host__ __device__ double operator()(double &x, double &y);
};

void normalize_wts(fvec_d &big_grid, int K, int G);

struct row_index: public thrust::unary_function<int, int>{
  int R, r;
  __host__ __device__ row_index(int _R, int _r): R(_R), r(_r){}
  __host__ __device__ int operator()(int &x){
    return R*x + r;
  }
};

typedef thrust::transform_iterator<row_index, countIter> rowIter;

rowIter getRowIter(int Rows, int row);

typedef thrust::permutation_iterator<fvec_d::iterator, rowIter> strideIter;

void gnl_multinomial(ivec_d &zeta, fvec_d &probs, curandState *states, int K, int G);

#endif
