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

typedef thrust::tuple<gRepEach<realIter>::iterator,
                      fvec_d::iterator,
                      ivec_d::iterator> compare_tup;

typedef thrust::zip_iterator<compare_tup> compare_zip;

typedef thrust::tuple<double &, double &, int &> compare_tup_el;

struct is_greater{
  __host__ __device__ void operator()(compare_tup_el Tup);
};

void gnl_multinomial(ivec_d &zeta, fvec_d &probs, curandState *states, int K, int G);

#endif
