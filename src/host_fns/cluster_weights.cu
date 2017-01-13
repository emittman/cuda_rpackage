#include "../header/cluster_probability.h"
#include "../header/iter_getter.h"

__host__ __device__ void clust_prob::operator()(weight_tup_el Tup){
    thrust::get<0>(Tup) = log(thrust::get<1>(Tup)) + 0.5 * n * log(thrust::get<2>(Tup)) - 0.5 * thrust::get<2>(Tup) * ( thrust::get<3>(Tup) - 2 * thrust::get<0>(Tup) + thrust::get<4>(Tup) );
}
  
void cluster_weights(fvec_d &big_grid, fvec_d &pi, fvec_d &tau2, fvec_d &yty, fvec_d &bxxb, int G, int V, int N, int K){
  gRepTimes<realIter>::iterator pi_iter = getGRepTimesIter(pi.begin(), pi.end(), K, 1);
  gRepTimes<realIter>::iterator tau_iter = getGRepTimesIter(tau2.begin(), tau2.end(), K, 1);
  gRepEach<realIter>::iterator yty_iter = getGRepEachIter(yty.begin(), yty.end(), K, 1);
  gRepTimes<realIter>::iterator bxxb_iter = getGRepTimesIter(bxxb.begin(), bxxb.end(), K, 1);
  weight_zip zipped = thrust::zip_iterator<weight_tup>(thrust::make_tuple(big_grid.begin(), pi_iter, tau_iter, yty_iter, bxxb_iter));
  clust_prob f(V*N);
  thrust::for_each(zipped, zipped + G*K, f);
}
