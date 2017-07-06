#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/for_each.h>
#include "../header/iter_getter.h"
#include "../header/quadform2.h"

//helper functions to get a constant iterator to a real-valued array
typedef thrust::permutation_iterator<realIter, thrust::constant_iterator<int> > gRepConst;

gRepConst getGRepConstIter(realIter begin, int index){
  thrust::constant_iterator<int> constIter = thrust::make_constant_iterator<int>(index);
  gRepConst iter = thrust::permutation_iterator<realIter, thrust::constant_iterator<int> >(begin, constIter);
  return iter;
}

//typedef for tuple type passed to functions
typedef thrust::tuple<double &, double &, double &> triple_tup;


__device__ void quadform(double *x, double *A, double *out, int V) {
  double tmp1 = 0;
  double tmp2 = 0;
  for(int i=0; i<V; i++){
    tmp1 = x[i];
    tmp2 += tmp1*tmp1*A[i * (V+1)];
    for(int j=i+1; j<V; j++){
      tmp2 += x[j]*tmp1*A[j+i*V];
    }
  }
  *out = tmp2;
}

struct quadform_funct{
  int V;
  
  __host__ __device__ quadform_funct(int V): V(V){}

  __host__ __device__ void operator()(triple_tup tup){
    double *x = thrust::raw_pointer_cast(&(thrust::get<0>(tup)));
    double *A = thrust::raw_pointer_cast(&(thrust::get<1>(tup)));
    double *out = thrust::raw_pointer_cast(&(thrust::get<2>(tup)));
    quadform(x, A, out, V);
  }
  
};

struct quadform_funct_simp{
  int V;
  fvec_d xtx;
  __host__ quadform_funct_simp(int _V, double * _xtx): V(_V){
    xtx.resize(V*V);
    thrust::copy(_xtx, _xtx + V*V, xtx.begin());
  }
  
  __host__ __device__ void operator()(triple_tup tup){
    double *x = thrust::raw_pointer_cast(&(thrust::get<0>(tup)));
    double *A = thrust::raw_pointer_cast(xtx.data());
    double *out = thrust::raw_pointer_cast(&(thrust::get<1>(tup)));
    quadform(x, A, out, V);
  }
  
};

typedef thrust::tuple<gRepTimes<realIter>::iterator, gRepEach<realIter>::iterator, realIter> quadTupGK;
typedef thrust::zip_iterator<quadTupGK> quadZipGK;

void quadform_multipleGK(fvec_d &beta, fvec_d &xtx, fvec_d &result, int G, int K, int V){
  
  quadform_funct f(V);

  gRepTimes<realIter>::iterator beta_skip = getGRepTimesIter(beta.begin(), beta.end(), K, V);
  gRepEach<realIter>::iterator xtx_skip = getGRepEachIter(xtx.begin(), xtx.end(), K, V*V);
  quadTupGK tup = thrust::tuple<gRepTimes<realIter>::iterator, gRepEach<realIter>::iterator, realIter>(beta_skip, xtx_skip, result.begin());
  quadZipGK zip = thrust::zip_iterator<quadTupGK>(tup);
  thrust::for_each(zip, zip + K*G, f);
}

typedef thrust::tuple<gRepTimes<realIter>::iterator, gRepConst, realIter> quadTupK;
typedef thrust::zip_iterator<quadTupK> quadZipK;

void quadform_multipleK(fvec_d &beta, fvec_d &xtx, fvec_d &result, int K, int V){
  
  quadform_funct_simp f(V, thrust::raw_pointer_cast(xtx.data()));

  gRepTimes<realIter>::iterator beta_skip = getGRepTimesIter(beta.begin(), beta.end(), K, V);
  gRepConst xtx_repeat = getGRepConstIter(xtx.begin(), 0);
  quadTupK tup = thrust::tuple<gRepTimes<realIter>::iterator, gRepConst, realIter>(beta_skip, xtx_repeat, result.begin());
  quadZipK zip = thrust::zip_iterator<quadTupK>(tup);
  thrust::for_each(zip, zip + K, f);
}

typedef thrust::tuple<gRepTimes<realIter>::iterator,gRepTimes<realIter>::iterator,realIter> quadTupMatch;
typedef thrust::zip_iterator<quadTupMatch> quadZipMatch;

void quadform_multipleMatch(fvec_d &beta, fvec_d &xtx, fvec_d &result, int K, int V){
  
  quadform_funct f(V);

  gRepTimes<realIter>::iterator beta_skip = getGRepTimesIter(beta.begin(), beta.end(), K, V);
  gRepTimes<realIter>::iterator xtx_skip = getGRepTimesIter(xtx.begin(), xtx.end(), K, V*V);
  quadTupMatch tup = thrust::tuple<gRepTimes<realIter>::iterator, gRepTimes<realIter>::iterator, realIter>(beta_skip, xtx_skip, result.begin());
  quadZipMatch zip = thrust::zip_iterator<quadTupMatch>(tup);
  thrust::for_each(zip, zip + K, f);
}
