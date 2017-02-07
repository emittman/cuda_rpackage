#include "iter_getter.h"

struct update_mean{
  int power;
  __host__ __device__ update_mean(int p): power(p){}
  template<typename Tup>
  __host__ __device__ operator()(Tup t){
    thrust::get<0>(t) = thrust::get<0>(t) + (pow(thrust::get<1>(t),power) - thrust::get<0>(t)) / thrust::get<2>(t);
  }
};

typedef thrust::tuple<realIter, realIter, thrust::constant_iterator<int>> update_tup;
typedef thrust::zip_iterator<update_tup> update_zip;

void update_running_means(fvec_d &means, fvec_d &new_obs, int length, int step, int power);