#include "thrust/device_vector.h"
#include "thrust/foreach.h""
#include "../header/beta_hat.h"
#include "cublas_v2.h"

typedef thrust::tuple<rowIter, rowIter> nrml_tuple;
typedef thrust::tuple<double &, double &> nrml_eltup;

struct solve_normal_eq{
  int dim;
  __host__ __device__ solve_normal_eq(int _dim): dim(_dim){}
  __host__ __device__ void operator()(nrml_eltup &Tup){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    int n=dim, lda=dim, incx=1;
    double *L = thrust::raw_pointer_cast(&(get<0>(tup)));
    double *x = thrust::raw_pointer_cast(&(get<1>(tup)));
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, L, bhat, incx);
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, L, bhat, incx);
  }
};

void beta_hat(fvec_d &chol_prec, fvec_d &xty, fvec &beta_hat, int K_occ, int V){
  thrust::copy(xty.begin(), xty.end(), beta_hat.begin());
  rowIter L_first = getRowIter(V*V, 0);
  rowIter xty_first = getRowIter(V, 0);
  strideIter L = thrust::permutation_iterator(chol_prec.begin(), L_first);
  strideIter x = thrust::permutation_iterator(beta_hat.begin(), xty_first);
  nrml_tuple my_tuple = thrust::make_tuple(L, x);
  thrust::zip_iterator<nrml_tuple> zipped = thrust::zip_iterator<nrml_tuple>(my_tuple);
  solve_normal_eq f(V);
  thrust::for_each(zipped, zipped + K_occ, f);
}


