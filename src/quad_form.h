#ifndef QUAD_FORM_H
#define QUAD_FORM_H

#include "thrust/device_vector.h"
#include "thrust/functional.h"
#include "thrust/transform.h"
#include "header/iter_getter.h"
#include "cublas_v2.h"


//helper functions to get a constant iterator to a real-valued array
typedef thrust::permutation_iterator<realIter, thrust::constant_iterator<int> > gRepConst;
gRepConst getGRepConstIter(realIter begin, int index){
  thrust::constant_iterator<int> constIter = thrust::make_constant_iterator<int>(index);
  gRepConst iter = thrust::permutation_iterator<realIter, thrust::constant_iterator<int> >(begin, constIter);
  return iter;
}

typedef thrust::tuple<gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator, gRepConst> qf_tup;
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

typedef thrust::device_vector<double> fvec;

//Compute t(x_i) %*% A %*% x_i where i=0, ..., n-1
void quad_form_multi(fvec &A, fvec &x, fvec &y, int n, int dim){
  if(A.size() != dim*dim) std::cout << "A.size() is not dim*dim!\n";
  if(y.size() != n) std::cout << "y.size() doesn't match inputs!";
  gRepTimes<realIter>::iterator x_strided = getGRepTimesIter(x.begin(), x.end(), n, dim);
  gRepConst A_repeat = getGRepConstIter(A.begin(), 0);
  fvec tmp(n*dim, 0.0);
  gRepTimes<realIter>::iterator tmp_strided = getGRepTimesIter(tmp.begin(), tmp.end(), n, dim);
  qf_tup my_tuple = thrust::tuple< gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator, gRepConst>(tmp_strided, x_strided, A_repeat);
  thrust::zip_iterator<qf_tup> zip_qf = thrust::zip_iterator<qf_tup>(my_tuple);
  quad_form f(dim);
  thrust::transform(zip_qf, zip_qf + n, y.begin(), f);
}


#endif
