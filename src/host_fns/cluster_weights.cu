#include "../header/cluster_probability.h"
#include "../header/iter_getter.h"

__host__ __device__ void clust_prob::operator()(weight_tup_el Tup){
    thrust::get<0>(Tup) = log(thrust::get<1>(Tup)) + n * 0.5 * log(thrust::get<2>(Tup)) + -0.5 * thrust::get<2>(Tup) * ( thrust::get<3>(Tup) - 2 * thrust::get<0>(Tup) + thrust::get<4>(Tup) );
}
  
void cluster_weights(fvec_d &big_grid, data_t &data, chain_t &chain){
  big_matrix_multiply(chain.beta, data.xty, big_grid, data.V, chain.K, data.V, data.G);
  fvec_d bxxb(chain.K);
  quad_form_multi(data.xtx, chain.beta, bxxb, chain.K, data.V);
  gRepTimes<realIter>::iterator pi_iter = getGRepTimesIter(chain.pi.begin(), chain.pi.end(), chain.K, 1);
  gRepTimes<realIter>::iterator tau_iter = getGRepTimesIter(chain.tau2.begin(), chain.tau2.end(), chain.K, 1);
  gRepEach<realIter>::iterator yty_iter = getGRepEachIter(data.yty.begin(), data.yty.end(), chain.K, 1);
  gRepTimes<realIter>::iterator bxxb_iter = getGRepTimesIter(bxxb.begin(), bxxb.end(), chain.K, 1);
  weight_zip zipped = thrust::zip_iterator<weight_tup>(thrust::make_tuple(big_grid.begin(), pi_iter, tau_iter, yty_iter, bxxb_iter));
  clust_prob f(data.N);
  thrust::for_each(zipped, zipped + data.G*chain.K, f);
}
