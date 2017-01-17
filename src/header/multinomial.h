#ifndef MULTINOM
#define MULTINOM

#include "curand_kernel.h"
#include "iter_getter.h"


struct log_sum_exp{
  __host__ __device__ double operator()(double &x, double &y);
};

void normalize_wts(fvec_d &big_grid, int K, int G);

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
