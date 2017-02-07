#include "../header/running_mean.h"

void update_running_means(fvec_d &means, fvec_d &new_obs, int length, int step, int power=1){
  thrust::constant_iterator<int> step_iter(step);
  update_tup tuple = thrust::tuple<realIter, realIter, thrust::constant_iterator<int>>(means.begin(), new_obs.begin(), step_iter);
  update_zip zip = thrust::zip_iterator<update_tup>(tuple);
  update_mean f(power);
  thrust::for_each(zip, zip + length, f);
}