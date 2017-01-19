#include "thrust/device_vector.h"
#include "thrust/for_each.h"
#include "../header/beta_hat.h"
#include "cublas_v2.h"

typedef thrust::tuple<strideIter, strideIter> nrml_tuple;
typedef thrust::tuple<double &, double &> nrml_eltup;

struct solve_normal_eq{
  int dim;
  __host__ __device__ solve_normal_eq(int _dim): dim(_dim){}
  __host__ __device__ void operator()(nrml_eltup &Tup){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    int n=dim, lda=dim, incx=1;
    double *L = thrust::raw_pointer_cast(&(thrust::get<0>(Tup)));
    double *x = thrust::raw_pointer_cast(&(thrust::get<1>(Tup)));
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, L, x, incx);
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, L, x, incx);
  }
};

void beta_hat(fvec_d &chol_prec, &beta_hat, int K_occ, int V){
  rowIter L_first = getRowIter(V*V, 0);
  rowIter xty_first = getRowIter(V, 0);
  strideIter L = thrust::permutation_iterator(chol_prec.begin(), L_first);
  strideIter x = thrust::permutation_iterator(beta_hat.begin(), xty_first);
  nrml_tuple my_tuple = thrust::make_tuple(L, x);
  thrust::zip_iterator<nrml_tuple> zipped = thrust::zip_iterator<nrml_tuple>(my_tuple);
  solve_normal_eq f(V);
  thrust::for_each(zipped, zipped + K_occ, f);
}


