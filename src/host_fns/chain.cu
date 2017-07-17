#include "../header/chain.h"
#include "../header/iter_getter.h"
#include "../header/transpose.h"
#include "../header/cluster_probability.h"
#include "../header/running_mean.h"
#include "../header/summary2.h"

data_t::data_t(double* _yty, double* _xty, double* _xtx, int _G, int _V, int _N, bool _voom): 
  G(_G), V(_V), N(_N), voom(_voom) {
  // 'carefully' copy data to device
  size_t yty_size = G;
  size_t xty_size = G*V;
  yty.resize(yty_size);
  xty.resize(xty_size);
  thrust::copy(_yty, _yty + yty_size, yty.begin());
  thrust::copy(_xty, _xty + xty_size, xty.begin()); 
  // store transpose of xty
  ytx.resize(V*G);
  transpose(xty.begin(), xty.end(), V, G, ytx.begin());
  
  // store transpose of xtx for reduction, i.e. xtx_0[0,0] + xtx_1[0,0]
  size_t xtx_size = (voom*(G-1)+1)*V*V;
  xtx.resize(xtx_size);
  thrust::copy(_xtx, _xtx + xtx_size, xtx.begin());
  txtx.resize(xtx_size);
  transpose(xtx.begin(), xtx.end(), V*V, voom*(G-1)+1, txtx.begin());
}

priors_t::priors_t(int _K, int _V, double* _mu0, double* _lambda2, double _a, double _b, double _A, double _B):
  K(_K), V(_V), a(_a), b(_b), A(_A), B(_B){
  size_t Vsize = _V;
  mu0.resize(Vsize);
  lambda2.resize(Vsize);
  thrust::copy(_mu0, _mu0 + Vsize, mu0.begin());
  thrust::copy(_lambda2, _lambda2 + Vsize, lambda2.begin());
}

samples_t::samples_t(int _n_save_g, int _n_save_P, int _G_save, int _K, int _V, int *idx, bool _alpha_fixed):
  n_save_g(_n_save_g), n_save_P(_n_save_P), step_g(0), step_P(0), G_save(_G_save), K(_K), V(_V),
  save_idx(idx, idx + _G_save), alpha_fixed(_alpha_fixed), save_beta(_n_save_g*_G_save*_V),
  save_tau2(_n_save_g*_G_save), save_P(_n_save_P*_K*(_V+2)), save_max_id(_n_save_g),
  save_num_occupied(_n_save_g), save_alpha(_n_save_g){}

void samples_t::write_g_samples(chain_t &chain, summary2 &smry){
  if(step_g < n_save_g){
    /* scatter cluster parameters by zeta for select genes */
    //get current map, save in tmpVec
    thrust::permutation_iterator<intIter, intIter> map_save_idx = thrust::permutation_iterator<intIter, intIter>(chain.zeta.begin(), save_idx.begin());
    ivec_d tmpVec(G_save);
    thrust::copy(map_save_idx, map_save_idx + G_save, tmpVec.begin());
    
    //copy betas (by column)
    SCIntIter beta_cpy = getSCIntIter(tmpVec.begin(), tmpVec.end(), chain.V);
    thrust::permutation_iterator<realIter, SCIntIter> betaI = thrust::permutation_iterator<realIter, SCIntIter>(chain.beta.begin(), beta_cpy);
    thrust::copy(betaI, betaI + G_save*V, save_beta.begin() + G_save*V*step_g);
    
    //copy tau2s
    thrust::permutation_iterator<realIter, intIter> tau2I = thrust::permutation_iterator<realIter, intIter>(chain.tau2.begin(), tmpVec.begin());
    thrust::copy(tau2I, tau2I + G_save, save_tau2.begin() + G_save*step_g);
    
    //save max_id
    thrust::copy(smry.occupied.end()-1, smry.occupied.end(), save_max_id.begin() + step_g);
    
    //save num_occupied
    save_num_occupied[step_g] = smry.num_occupied;
    
    step_g += 1;
  }
  else std::cout << "step_g >= n_save_g!";
}

void samples_t::write_P_samples(chain_t &chain){
  if(step_P < n_save_P){
    // (number of clusters) * (dimension of beta[k] + dimension of tau2[k] + dimension of pi[k])
    int iter_size = K*(V+2);
    // copy clusters to save_P
    thrust::host_vector<double>::iterator save_P_iter = save_P.begin() + iter_size * step_P;
    thrust::copy(chain.pi.begin(), chain.pi.end(), save_P_iter);
    transpose<realIter, thrust::host_vector<double>::iterator>(chain.beta.begin(), chain.beta.end(), chain.V, chain.K, save_P_iter + K);
    thrust::copy(chain.tau2.begin(), chain.tau2.end(), save_P_iter + K*(V+1));
    step_P = step_P + 1;
  } else std::cout << "step_P >= n_save_P!";
}

void chain_t::update_probabilities(int step){
  /* grid = C %*% beta, Igrid = I(grid>0), resultK = reduce_by_key(P_rowid(column), Igrid, minimum)*/
  fvec_d grid(P*K);
  ivec_d Igrid(P*K);
  fvec_d resultK(n_hyp*K);
  
  big_matrix_multiply(C, beta, grid, V, P, V, K);
  
  thrust::transform(grid.begin(), grid.end(), Igrid.begin(), thrust::placeholders::_1 > 0);
  
  RSIntIter group_by_hyp = getRSIntIter(C_rowid.begin(), C_rowid.end(), n_hyp);
  thrust::reduce_by_key(group_by_hyp, group_by_hyp + P*K, Igrid.begin(), thrust::make_discard_iterator(), resultK.begin(), thrust::equal_to<int>(), thrust::minimum<int>());
  
  /* map by zeta*/
  fvec_d resultG(n_hyp*G);
  SCIntIter kcols_to_gcols = getSCIntIter(zeta.begin(), zeta.end(), n_hyp);
  thrust::permutation_iterator<realIter, SCIntIter> map_result = thrust::permutation_iterator<realIter, SCIntIter>(resultK.begin(), kcols_to_gcols);
  thrust::copy(map_result, map_result + n_hyp*G, resultG.begin());
  update_running_means(probs, resultG, n_hyp*G, step, 1);
}

struct inv_sqrt: thrust::unary_function<double, double>{
  __host__ __device__ double operator()(double x){
    return 1/sqrt(x);
  }
};

void chain_t::update_means(int step){
  fvec_d beta_g(G*V);
  fvec_d sigma_g(G);
  SCIntIter map_zeta = getSCIntIter(zeta.begin(), zeta.end(), V);
  thrust::permutation_iterator<realIter, SCIntIter> map_beta = thrust::permutation_iterator<realIter, SCIntIter>(beta.begin(), map_zeta);
  thrust::permutation_iterator<realIter, intIter> map_tau2 = thrust::permutation_iterator<realIter, intIter>(tau2.begin(), zeta.begin());
  thrust::copy(map_beta, map_beta+G*V, beta_g.begin());
  thrust::copy(map_tau2, map_tau2+G, sigma_g.begin());
  thrust::transform(sigma_g.begin(), sigma_g.end(), sigma_g.begin(), inv_sqrt());
  update_running_means(means_betas, beta_g, G*V, step, 1);
  update_running_means(meansquares_betas, beta_g, G*V, step, 2);
  update_running_means(means_sigmas, sigma_g, G, step, 1);
  update_running_means(meansquares_sigmas, sigma_g, G, step, 2);
}

