#include "../header/multi_dot_product.h"
typedef thrust::tuple<strideIter, strideIter, realIter> dot_tup;
typedef thrust::tuple<double &, double &, double &> dot_eltup;

struct dot_prod {
  int dim;
  __host__ __device__ dot_prod(int _dim): dim(_dim){}
  __host__ __device__ void operator()(dot_eltup &Tup){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    int incx=1, incy=1;
    double *x = thrust::raw_pointer_cast(&(thrust::get<0>(Tup)));
    double *y = thrust::raw_pointer_cast(&(thrust::get<1>(Tup)));
    double *z = thrust::raw_pointer_cast(&(thrust::get<2>(Tup)));
    cublasDdot(handle, dim, x, incx, y, incy, z);
    cublasDestroy_v2(handle);
  }
};

void multi_dot_prod(fvec_d &x, fvec_d &y, fvec_d &z, int dim, int n){
  rowIter x_first = getRowIter(dim, 0);
  rowIter y_first = getRowIter(dim, 0);
  strideIter x = thrust::permutation_iterator<realIter, rowIter>(x.begin(), x_first);
  strideIter y = thrust::permutation_iterator<realIter, rowIter>(y.begin(), y_first);
  dot_tup my_tuple = thrust::make_tuple(x, y, z.begin());
  thrust::zip_iterator<dot_tuple> zipped = thrust::zip_iterator<dot_tuple>(my_tuple);
  dot_prod f(dim);
  thrust::for_each(zipped, zipped + n, f);
}
