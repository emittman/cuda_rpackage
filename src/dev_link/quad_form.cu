#include "thrust/device_vector.h"
#include "thrust/functional.h"
#include "thrust/transform.h"
#include "../header/iter_getter.h"
#include "cublas_v2.h"

typedef thrust::tuple<gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator, gConst<realIter>::iterator> qf_tup;
typedef thrust::tuple<gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator> qf_tup2;
typedef thrust::tuple<double &, double &, double &> ftrip;

struct quad_form: thrust::unary_function<ftrip, double>{
  int dim;
  __host__ __device__ quad_form(int dim): dim(dim){}
  __host__ __device__ double operator()(ftrip &trip){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    int incx=1, incy=1, n = dim, lda=dim;
    double alpha=1, beta=0;
    double *y = thrust::raw_pointer_cast(&(thrust::get<0>(trip)));
    double *x = thrust::raw_pointer_cast(&(thrust::get<1>(trip)));
    double *A = thrust::raw_pointer_cast(&(thrust::get<2>(trip)));
    double result;
    cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER, n, &alpha, A, lda, x, incx, &beta, y, incy);
    cublasDdot(handle, n, x, incx, y, incy, &result);
    cublasDestroy_v2(handle);
    return result;
  }
};


//Compute t(x_i) %*% A %*% x_i where i=0, ..., n-1
void quad_form_multi(fvec_d &A, fvec_d &x, fvec_d &y, int n, int dim, bool fixed_A){
  if(!fixed_A){
    if(A.size() != n*dim*dim)
    std::cout << "in quad_form_multi:\t fixed_A is false, but A.size() != n*dim*dim\n";
  } else if(A.size() != dim*dim){
    std::cout << "in quad_form_multi:\t fixed_A is true, but A.size() != dim*dim\n";
  }
  if(y.size() != n) std::cout << "y.size() doesn't match inputs!";
  gRepTimes<realIter>::iterator x_strided = getGRepTimesIter(x.begin(), x.end(), n, dim);
  fvec_d tmp(n*dim, 0.0);
  gRepTimes<realIter>::iterator tmp_strided = getGRepTimesIter(tmp.begin(), tmp.end(), n, dim);

  if(fixed_A){
    // Inner matrix is same for all g = 1:G
    gConst<realIter>::iterator A_repeat = getGConstIter(A.begin(), 0);
    qf_tup my_tuple = thrust::tuple< gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator, gConst<realIter>::iterator>(tmp_strided, x_strided, A_repeat);
    thrust::zip_iterator<qf_tup> zip_qf = thrust::zip_iterator<qf_tup>(my_tuple);
    quad_form f(dim);
    thrust::transform(zip_qf, zip_qf + n, y.begin(), f);
    
  } else{
    // Inner matrices are different for all g = 1:G
    gRepTimes<realIter>::iterator A_strided = getGRepTimesIter(A.begin(), A.end(), n, dim*dim);
    qf_tup2 my_tuple = thrust::tuple< gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator>(tmp_strided, x_strided, A_strided);
    thrust::zip_iterator<qf_tup2> zip_qf = thrust::zip_iterator<qf_tup2>(my_tuple);
    quad_form f(dim);
    thrust::transform(zip_qf, zip_qf + n, y.begin(), f);
    
  }
}


