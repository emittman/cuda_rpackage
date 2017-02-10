#include "../header/chain.h"
#include "../header/iter_getter.h"
#include "../header/transpose.h"
#include "../header/cluster_probability.h"
#include "../header/running_mean.h"


data_t::data_t(double* _yty, double* _xty, double* _xtx, int _G, int _V, int _N): 
    yty(_yty, _yty + _G), xty(_xty, _xty + _G*_V), xtx(_xtx, _xtx + _V*_V), G(_G), V(_V), N(_N) {
    ytx = fvec_d(G*V);
    transpose(xty.begin(), xty.end(), V, G, ytx.begin());
}

samples_t::samples_t(int _n_iter, int _K_save, int _V, int *idx):
    n_iter(_n_iter), step(0), K_save(_K_save), V(_V),
    save_idx(idx, idx + K_save), save_beta(iter*K_save*V), save_tau2(iter*K_save), save_pi(iter*K_save), beta_iter(getSCIntIter(save_idx.begin(), save_idx.end(), V)){}

void samples_t::write_samples(chain_t &chain){
  if(step < n_iter){
    thrust::permutation_iterator<realIter, SCIntIter> betaI = thrust::permutation_iterator<realIter, SCIntIter>(chain.beta.begin(), beta_iter);
    thrust::copy(betaI, betaI + K_save*V, save_beta.begin() + K_save*V*step);
    thrust::permutation_iterator<realIter, intIter> tau2I = thrust::permutation_iterator<realIter, intIter>(chain.tau2.begin(), save_idx.begin());
    thrust::copy(tau2I, tau2I + K_save, save_tau2.begin() + K_save*step);
    thrust::permutation_iterator<realIter, intIter> piI = thrust::permutation_iterator<realIter, intIter>(chain.pi.begin(), save_idx.begin());
    thrust::copy(piI, piI + K_save, save_pi.begin() + K_save*step);
    step += 1;
  }
  else std::cout << "step >= n_iter!";
}

void chain_t::update_probabilities(int step){
  /* multiply C %*% beta, eval if > 0, resultK = (min(Col) == 1)[all true]*/
  fvec_d grid(P*K);
  ivec_d Igrid(P*K);
  fvec_d resultK(K);
  big_matrix_multiply(C, beta, grid, V, P, V, K);
  thrust::transform(grid.begin(), grid.end(), Igrid.begin(), thrust::placeholders::_1 > 0);
  repEachIter colIt = getRepEachIter(P, 1);
  thrust::reduce_by_key(colIt, colIt + P*K, Igrid.begin(), thrust::make_discard_iterator(), resultK.begin(), thrust::equal_to<int>(), thrust::minimum<int>());
  /* map by zeta*/
  fvec_d resultG(G);
  thrust::permutation_iterator<realIter, intIter> map_result = thrust::permutation_iterator<realIter, intIter>(resultK.begin(), zeta.begin());
  thrust::copy(map_result, map_result+G, resultG.begin());
  update_running_means(probs, resultG, G, step, 1);
}

void chain_t::update_means(int step){
  fvec_d beta_g(G*V);
  SCIntIter map_zeta = getSCIntIter(zeta.begin(), zeta.end(), V);
  thrust::permutation_iterator<realIter, SCIntIter> map_beta = thrust::permutation_iterator<realIter, SCIntIter>(beta.begin(), map_zeta);
  thrust::copy(map_beta, map_beta+G*V, beta_g.begin());
  update_running_means(means, beta_g, G*V, step, 1);
  update_running_means(meansquares, beta_g, G*V, step, 2);
}

