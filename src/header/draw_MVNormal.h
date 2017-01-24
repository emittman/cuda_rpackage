#ifndef DRAW_MVN
#define DRAW_MVN

#include "distribution.h"
#include "iter_getter.h"
#include "summary2.h"

struct scale_vec{
  int dim;
  __host__ __device__ scale_vec(int _dim): dim(_dim) {}
  
  template <typename T>
  __host__ __device__ void operator()(T tup){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    int n=dim, lda=dim, incx=1;
    double *z = thrust::raw_pointer_cast(&(get<0>(tup)));
    double *L = thrust::raw_pointer_cast(&(get<1>(tup)));
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, L, z, incx); // t(L)^{-1} %*% t(t(L)^{-1}) = (L%*%t(L))^{-1}
    cublasDestroy(handle);
  }
};

struct mult_scalar_by_sqrt{
  template <typename T>
  __host__ __device__ void operator()(T tup){
    thrust::get<0>(tup) *= sqrt(thrust::get<1>(tup));
  }
};

void summary2::draw_MVNormal(curandState *states, fvec_d &beta_hat, fvec_d &chol_prec, fvec_d &beta, priors_t &priors);


#endif