#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include "iterator2.h"

__device__ void cholesky(double *A, int n) {
  for (int j = 0; j < n; ++j){
    for (int k = 0; k < j; ++k) {
      double tmp = A[j+k*n];
      for (int i = j; i < n; ++i){
        A[i+j*n] = A[i+j*n] - A[i+k*n]*tmp;
      }
    }
    double tmp = sqrt(A[j+j*n]);
    A[j+j*n] = tmp;
    for(int k = j+1; k<n; ++k)
      A[k+j*n] = A[k+j*n]/tmp;
  }
}

struct chol{
  int dim;
  
  __host__ __device__ chol(int dim): dim(dim){}
  
  __host__ __device__ void operator()(double &first){
    double *A = thrust::raw_pointer_cast(& first);
    cholesky(A, dim);
  }
  
};

__host__ void chol_multiple(thrust::device_vector<double> &dvec, int dim, int n){
  
  chol f(dim);
  typename gRepTimes<double>::iterator mat_first = getGRepTimes(dvec.begin(), dvec.end(), n, dim*dim);
  thrust::for_each(mat_first, mat_first+n, f);
  
}