#include "../header/gibbs.h"
#include <thrust/scan.h>
#include <thrust/transform_scan.h>

struct log_1m {
  __host__ __device__ double operator()(double &x){
    return log(1-x);
  }
};

struct exp_log_plus {
  __host__ __device__ double operator()(double &x, double &y){
    return exp(log(x) + y);
  }
};

struct modify_gamma_par {
  template<typename T>
  __host__ __device__ void operator()(T tup){
    thrust::get<0>(tup) = thrust::get<0>(tup) + 1.0/ 2.0 * thrust::get<1>(tup);
  }
};

void draw_MVNormal(curandState *states, fvec_d &beta_hat, fvec_d &chol_prec, fvec_d &beta, priors_t &priors, summary2 &smry){
  int K = smry.K;
  int V = smry.V;
  //replace current beta with standard normal draws
  getNormal<<<K, V>>>(states, thrust::raw_pointer_cast(beta.data()));
  
  //std::cout << "N(0,1) draws:\n";
  //printVec(beta, V, K);
  
  //scale occupied betas by t(chol_prec)^-1
  scale_chol_inv(chol_prec, beta, smry.occupied, smry.num_occupied, V);

  //std::cout << "scaled draws:\n";
  //printVec(beta, V, K);
  
  //typedef: iterate along select columns of matrix of doubles
  typedef thrust::permutation_iterator<realIter, SCIntIter> gSCIter;
  
  //need access to occupied betas
  SCIntIter occ_idx = getSCIntIter(smry.occupied.begin(), smry.occupied.end(), V);
  gSCIter betaOcc = thrust::permutation_iterator<realIter, SCIntIter>(beta.begin(), occ_idx);
  
  //shift occupied betas by beta_hat
  thrust::transform(beta_hat.begin(), beta_hat.end(), betaOcc, betaOcc, thrust::plus<double>());
  
  //std::cout << "occupied draws:\n";
  //printVec(beta, V, K);
  
  //now, access to unoccupied betas
  SCIntIter unocc_idx = getSCIntIter(smry.unoccupied.begin(), smry.unoccupied.end(), V);
  gSCIter betaUnocc = thrust::permutation_iterator<realIter, SCIntIter>(beta.begin(), unocc_idx);
  
  //repeat prior var. and mean
  gRepTimes<realIter>::iterator prior_vars = getGRepTimesIter(priors.lambda2.begin(), priors.lambda2.end(), V);
  gRepTimes<realIter>::iterator prior_mean = getGRepTimesIter(priors.mu0.begin(), priors.mu0.end(), V);
  
  typedef thrust::tuple<gSCIter, gRepTimes<realIter>::iterator> linTransSomeBeta_tup;
  typedef thrust::zip_iterator<linTransSomeBeta_tup> linTransSomeBeta_zip;
  
  int num_unoccupied = K - smry.num_occupied;
  
  //scale by prior sd
  linTransSomeBeta_tup scale_tup2 = thrust::tuple<gSCIter, gRepTimes<realIter>::iterator>(betaUnocc, prior_vars);
  linTransSomeBeta_zip scale_zip2 = thrust::zip_iterator<linTransSomeBeta_tup>(scale_tup2);
  mult_scalar_by_sqrt f2;
  thrust::for_each(scale_zip2, scale_zip2 + num_unoccupied*V, f2);
  
  //std::cout << "unoccupied are scaled now:\n";
  //printVec(beta, V, K);
  //shift by prior mean
  thrust::transform(prior_mean, prior_mean + num_unoccupied*V, betaUnocc, betaUnocc, thrust::plus<double>());
  //std::cout << "and shifted (final draws):\n";
  //printVec(beta, V, K);
}

void draw_tau2(curandState *states, chain_t &chain, priors_t &priors, data_t &data, summary2 &smry){
  fvec_d sse(smry.num_occupied);
  int K = chain.K;
  smry.sumSqErr(sse, chain.beta, data.xtx);
  //std::cout << "sse:\n";
  //printVec(sse, K, 1);
  fvec_d a_d(K, priors.a);
  fvec_d b_d(K, priors.b);
  //std::cout << "a_d filled:\n";
  //printVec(a_d, K, 1);
  // modify gamma parameters for occupied clusters
  typedef thrust::tuple<realIter, intIter> tuple1;
  typedef thrust::zip_iterator<tuple1> zip1;
  tuple1 tup1 = thrust::tuple<realIter, intIter>(a_d.begin(), smry.Mk.begin());
  zip1 zp1 = thrust::zip_iterator<tuple1>(tup1);
  
  modify_gamma_par f;
  thrust::for_each(zp1, zp1 + K, f);
  
  //std::cout << "a transformed:\n";
  //printVec(a_d, K, 1);
  
  typedef thrust::permutation_iterator<realIter, intIter> FltPermIter;
  FltPermIter b_occ = thrust::permutation_iterator<realIter, intIter>(b_d.begin(), smry.occupied.begin());
  typedef thrust::tuple<FltPermIter, realIter> tuple2;
  typedef thrust::zip_iterator<tuple2> zip2;
  tuple2 tup2 = thrust::tuple<FltPermIter, realIter>(b_occ, sse.begin());
  zip2 zp2 = thrust::zip_iterator<tuple2>(tup2);
  thrust::for_each(zp2, zp2 + K, modify_gamma_par());
  
  //std::cout << "b transformed:\n";
  //printVec(b_d, K, 1);
  // raw pointers
  double *tau2_ptr = thrust::raw_pointer_cast(chain.tau2.data());
  double *a_ptr = thrust::raw_pointer_cast(a_d.data());
  double *b_ptr = thrust::raw_pointer_cast(b_d.data());
  
  //generate
  getGamma<<<K, 1>>>(states, a_ptr, b_ptr, tau2_ptr);
}

void draw_pi(curandState *states, chain_t &chain, priors_t &priors, summary2 &summary){
  int K = priors.K;
  fvec_d Tk(K);
  fvec_d Mkp1(K);
  fvec_d Vk(K, 1.0);
  //std::cout << "Tk init:\n";
  //printVec(Tk, K, 1);
  thrust::exclusive_scan(summary.Mk.rbegin(), summary.Mk.rend(), Tk.rbegin());
  //std::cout << "Tk filled:\n";
  //printVec(Tk, K, 1);
  thrust::transform(Tk.begin(), Tk.end(), Tk.begin(), thrust::placeholders::_1 + priors.alpha);
  //std::cout <<"Tk transformed";
  //printVec(Tk, K, 1);
  thrust::transform(summary.Mk.begin(), summary.Mk.end(), Mkp1.begin(), thrust::placeholders::_1 + 1.0);
  getBeta<<<K-1, 1>>>(states, thrust::raw_pointer_cast(Mkp1.data()),
                    thrust::raw_pointer_cast(Tk.data()),
                    thrust::raw_pointer_cast(Vk.data()));
  //std::cout <<"Vk:\n";
  //printVec(Vk, K, 1);
  fvec_d Ck(K, 0.0);
  transform_inclusive_scan(Vk.begin(), Vk.end()-1, Ck.begin()+1, log_1m(), thrust::plus<double>());
  //std::cout << "Ck:\n";
  //printVec(Ck, K, 1);
  transform(Vk.begin(), Vk.end(), Ck.begin(), chain.pi.begin(), exp_log_plus());
  //std::cout << "pi:\n";
  //printVec(chain.pi, K, 1);
}

void draw_zeta(curandState *states, data_t &data, chain_t &chain, priors_t &priors){
  fvec_d grid(data.G*priors.K);
  cluster_weights(grid, data, chain);
  //std::cout << "grid:\n";
  //printVec(grid, priors.K, data.G);
  gnl_multinomial(chain.zeta, grid, states, priors.K, data.G);
  //std::cout << "(inside draw_zeta) zeta:\n";
  //printVec(chain.zeta, data.G, 1);
}

void draw_beta(curandState *states, data_t &data, chain_t &chain, priors_t &priors, summary2 &smry){
  fvec_d prec(smry.num_occupied * data.V * data.V);
  fvec_d betahat(smry.num_occupied * data.V);
  //get cluster (inv)scales
  construct_prec(prec.begin(), prec.end(), priors.lambda2.begin(), priors.lambda2.end(), chain.tau2.begin(), chain.tau2.end(), smry.Mk.begin(), smry.Mk.end(), data.xtx.begin(), data.xtx.end(), smry.num_occupied, data.V);
  realIter prec_begin = prec.begin();
  realIter prec_end = prec.end();
  chol_multiple(prec_begin, prec_end, data.V, smry.num_occupied);
  //get cluster locations
  thrust::device_ptr<double> xty_ptr = &smry.xty_sums[0];
  thrust::copy(xty_ptr, xty_ptr + smry.num_occupied*data.V, betahat.begin());
  beta_hat(prec, betahat, smry.num_occupied, data.V);
  std::cout << "beta_hat:\n";
  printVec(betahat, data.V, smry.num_occupied);
  draw_MVNormal(states, betahat, prec, chain.beta, priors, smry);
}


