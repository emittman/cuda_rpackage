#include "../header/beta_hat.h"


typedef thrust::tuple<strideIter, strideIter> nrml_tuple;
typedef thrust::tuple<double &, double &> nrml_eltup;

struct solve_normal_eq {
  int dim;
  __host__ __device__ solve_normal_eq(int _dim): dim(_dim){}
  template <typename T>
  __host__ __device__ void operator()(T Tup){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    int n=dim, lda=dim, incx=1;
    double *L = thrust::raw_pointer_cast(&(thrust::get<0>(Tup)));
    double *x = thrust::raw_pointer_cast(&(thrust::get<1>(Tup)));
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, L, lda, x, incx);
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, L, lda, x, incx);
    cublasDestroy_v2(handle);
  }
};

struct left_mult_chol_inv{
  int dim;
  __host__ __device__ left_mult_chol_inv(int _dim): dim(_dim) {}
  
  template <typename T>
  __host__ __device__ void operator()(T tup){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    int n=dim, lda=dim, incx=1;
    double *z = thrust::raw_pointer_cast(&(thrust::get<0>(tup)));
    double *L = thrust::raw_pointer_cast(&(thrust::get<1>(tup)));
    // t(L)^{-1} %*% t(t(L)^{-1}) = (L%*%t(L))^{-1}
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, L, lda, z, incx);
    cublasDestroy(handle);
  }
};

void beta_hat(fvec_d &chol_prec, fvec_d &beta_hat, int K, int V){
  rowIter L_first = getRowIter(V*V, 0);
  rowIter xty_first = getRowIter(V, 0);
  strideIter L = thrust::permutation_iterator<realIter, rowIter>(chol_prec.begin(), L_first);
  strideIter x = thrust::permutation_iterator<realIter, rowIter>(beta_hat.begin(), xty_first);
  nrml_tuple my_tuple = thrust::make_tuple(L, x);
  thrust::zip_iterator<nrml_tuple> zipped = thrust::zip_iterator<nrml_tuple>(my_tuple);
  solve_normal_eq f(V);
  thrust::for_each(zipped, zipped + K, f);
  //std::cout << "betahat:\n";
  //printVec(beta_hat, V, K);
}

void scale_chol_inv(fvec_d &chol_prec, fvec_d &z, int n, int dim){
  typedef thrust::tuple<strideIter, strideIter> scl_z_tup;
  typedef thrust::zip_iterator<scl_z_tup> scl_z_zip;
  //need access to first elems of chols and occ. betas
  rowIter strides_z = getRowIter(dim, 0);
  rowIter strides_L = getRowIter(dim*dim, 0);
  strideIter z_first = thrust::permutation_iterator<realIter, rowIter>(z.begin(), strides_z);
  strideIter L_first = thrust::permutation_iterator<realIter, rowIter>(chol_prec.begin(), strides_L);
  scl_z_zip scale_zip = thrust::zip_iterator<scl_z_tup>(thrust::make_tuple(z_first, L_first));
  left_mult_chol_inv f(dim);
  thrust::for_each(scale_zip, scale_zip + n, f);
}
