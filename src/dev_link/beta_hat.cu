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
    double *z = thrust::raw_pointer_cast(&(get<0>(tup)));
    double *L = thrust::raw_pointer_cast(&(get<1>(tup)));
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, L, z, incx); // t(L)^{-1} %*% t(t(L)^{-1}) = (L%*%t(L))^{-1}
    cublasDestroy(handle);
  }
};

void beta_hat(fvec_d &chol_prec, fvec_d &beta_hat, int K_occ, int V){
  rowIter L_first = getRowIter(V*V, 0);
  rowIter xty_first = getRowIter(V, 0);
  strideIter L = thrust::permutation_iterator<realIter, rowIter>(chol_prec.begin(), L_first);
  strideIter x = thrust::permutation_iterator<realIter, rowIter>(beta_hat.begin(), xty_first);
  nrml_tuple my_tuple = thrust::make_tuple(L, x);
  thrust::zip_iterator<nrml_tuple> zipped = thrust::zip_iterator<nrml_tuple>(my_tuple);
  solve_normal_eq f(V);
  thrust::for_each(zipped, zipped + K_occ, f);
}

void scale_chol_inv(fvec_d &chol_prec, fvec_d &x, ivec_d &idx, int len_idx, int dim){
  typedef thrust::tuple<gSFRIter<realIter>::iterator, strideIter> scl_sel_x_tup;
  typedef thrust::zip_iterator<scaleSomeBeta_tup> scl_sel_x_zip;
  //need access to first elems of chols and occ. betas
  gSFRIter<realIter>::iterator sel_x_first = getGSFRIter(x.begin(), x.end(), idx, dim);
  rowIter strides_idx = getRowIter(dim*dim, 0);
  strideIter L_first = thrust::permutation_iterator<realIter, rowIter>(chol_prec.begin(), strides_idx);
  scl_sel_x_tup scale_tup = thrust::make_tuple(sel_x_first, L_first);
  scl_sel_x_zip scale_zip = thrust::zip_iterator<scl_sel_x_tup>(scale_tup);
  left_mult_chol_inv f(V);
  
  //scale x[idx]
  thrust::for_each(scale_zip, scale_zip + len_idx, f);
}
