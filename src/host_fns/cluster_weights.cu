#include "../header/cluster_probability.h"
#include "../header/iter_getter.h"

__host__ __device__ void clust_prob::operator()(weight_tup_el Tup){
    thrust::get<0>(Tup) = thrust::get<1>(Tup) + n * 0.5 * log(thrust::get<2>(Tup)) + -0.5 * thrust::get<2>(Tup) * ( thrust::get<3>(Tup) - 2 * thrust::get<0>(Tup) + thrust::get<4>(Tup) );
}
  
void cluster_weights_voom(fvec_d &big_grid, data_t &data, chain_t &chain, int verbose=0){
  if(verbose>0){
    std::cout << "big matrix multiply...\n";
  }
  big_matrix_multiply(chain.beta, data.xty, big_grid, data.V, chain.K, data.V, data.G);

  if(verbose>0){
    std::cout << "quadratic form btxtxb...\n";
  }
  fvec_d bxxb(chain.K*chain.G);
  quadform_multipleGK(chain.beta, data.xtx, bxxb, chain.G, chain.K, chain.V);

  gRepTimes<realIter>::iterator pi_iter = getGRepTimesIter(chain.pi.begin(), chain.pi.end(), chain.K, 1);
  gRepTimes<realIter>::iterator tau_iter = getGRepTimesIter(chain.tau2.begin(), chain.tau2.end(), chain.K, 1);
  gRepEach<realIter>::iterator yty_iter = getGRepEachIter(data.yty.begin(), data.yty.end(), chain.K, 1);
  realIter bxxb_iter = bxxb.begin();
  weight_tup_voom tup = thrust::tuple<realIter, gRepTimes<realIter>::iterator,gRepTimes<realIter>::iterator,
                                      gRepEach<realIter>::iterator, realIter>(
                                      big_grid.begin(), pi_iter, tau_iter, yty_iter, bxxb_iter);
  weight_zip_voom zipped = thrust::zip_iterator<weight_tup_voom>(tup);
  //weight_zip_voom zipped = thrust::zip_iterator<weight_tup_voom>(thrust::make_tuple(big_grid.begin(), pi_iter, tau_iter, yty_iter, bxxb_iter));
  clust_prob f(data.N);
  if(verbose>0){
    std::cout << "final weight computation (sum)...\n";
  }
  thrust::for_each(zipped, zipped + data.G*chain.K, f);
}

void cluster_weights_no_voom(fvec_d &big_grid, data_t &data, chain_t &chain, int verbose=0){
  if(verbose>0){
    std::cout << "big matrix multiply...\n";
  }
  big_matrix_multiply(chain.beta, data.xty, big_grid, data.V, chain.K, data.V, data.G);
  if(verbose>0){
    printVec(big_grid, chain.K, chain.G);
  }
  if(verbose>0){
    std::cout << "quadratic form btxtxb...\n";
    
  }
  
  fvec_d bxxb(chain.K);
  quadform_multipleK(chain.beta, data.xtx, bxxb, chain.K, chain.V);
  
  if(verbose>0){
    printVec(bxxb, chain.K, 1);
  }
  
  gRepTimes<realIter>::iterator pi_iter = getGRepTimesIter(chain.pi.begin(), chain.pi.end(), chain.K, 1);
  gRepTimes<realIter>::iterator tau_iter = getGRepTimesIter(chain.tau2.begin(), chain.tau2.end(), chain.K, 1);
  gRepEach<realIter>::iterator yty_iter = getGRepEachIter(data.yty.begin(), data.yty.end(), chain.K, 1);
  gRepTimes<realIter>::iterator bxxb_iter = getGRepTimesIter(bxxb.begin(), bxxb.end(), chain.K, 1);
  weight_zip_no_voom zipped = thrust::zip_iterator<weight_tup_no_voom>(thrust::make_tuple(big_grid.begin(), pi_iter, tau_iter, yty_iter, bxxb_iter));
  clust_prob f(data.N);
  if(verbose>0){
    std::cout << "final weight computation (sum)...\n";
  }
  thrust::for_each(zipped, zipped + data.G*chain.K, f);
}
