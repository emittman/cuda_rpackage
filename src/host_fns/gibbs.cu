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
  double N;
  modify_gamma_par(double _N): N(_N){}
  template<typename T>
  __host__ __device__ void operator()(T tup){
    thrust::get<0>(tup) = thrust::get<0>(tup) + 1.0/ 2.0 * thrust::get<1>(tup) * N;
  }
};

void draw_MVNormal(curandState *states, fvec_d &beta_hat, fvec_d &chol_prec, fvec_d &beta, priors_t &priors, int verbose = 0){
  //no longer should be passing summary2!
  int K = priors.K;
  int V = priors.V;
  //replace current beta with standard normal draws
  getNormal<<<K, V>>>(states, thrust::raw_pointer_cast(beta.data()));
  
  if(verbose > 1){
    std::cout << "N(0,1) draws:\n";
    printVec(beta, V, K);
  }
  
  //scale occupied betas by t(chol_prec)^-1
  scale_chol_inv(chol_prec, beta, K, V);

  if(verbose > 1){
    std::cout << "scaled draws:\n";
    printVec(beta, V, K);
  }
  
  //shift betas by beta_hat
  thrust::transform(beta_hat.begin(), beta_hat.end(), beta.begin(), beta.begin(), thrust::plus<double>());
  
  if(verbose > 1){
    std::cout << "beta draws:\n";
    printVec(beta, V, K);
  }
}

void draw_tau2(curandState *states, chain_t &chain, priors_t &priors, data_t &data, summary2 &smry, int verbose=0){
  fvec_d sse(smry.num_occupied);
  int K = chain.K;
  smry.sumSqErr(sse, chain.beta, data.xtx, --verbose);
  if(verbose > 1){
    std::cout << "sse:\n";
    printVec(sse, smry.num_occupied, 1);
  }
  fvec_d a_d(K, priors.a);
  fvec_d b_d(K, priors.b);
  if(verbose > 1){
    std::cout << "a_d filled:\n";
    printVec(a_d, K, 1);
  }
  // modify gamma parameters for occupied clusters
  typedef thrust::tuple<realIter, intIter> tuple1;
  typedef thrust::zip_iterator<tuple1> zip1;
  tuple1 tup1 = thrust::tuple<realIter, intIter>(a_d.begin(), smry.Mk.begin());
  zip1 zp1 = thrust::zip_iterator<tuple1>(tup1);
  modify_gamma_par f1(data.N);
  thrust::for_each(zp1, zp1 + K, f1);
  
  if(verbose > 1){
    std::cout << "a transformed:\n";
    printVec(a_d, K, 1);
  }
  
  typedef thrust::permutation_iterator<realIter, intIter> FltPermIter;
  FltPermIter b_occ = thrust::permutation_iterator<realIter, intIter>(b_d.begin(), smry.occupied.begin());
  typedef thrust::tuple<FltPermIter, realIter> tuple2;
  typedef thrust::zip_iterator<tuple2> zip2;
  tuple2 tup2 = thrust::tuple<FltPermIter, realIter>(b_occ, sse.begin());
  zip2 zp2 = thrust::zip_iterator<tuple2>(tup2);
  modify_gamma_par f2(1.0);
  if(verbose > 1){
    std::cout << "b filled:\n";
    printVec(b_d, K, 1);
  }
  thrust::for_each(zp2, zp2 + smry.num_occupied, f2);

  if(verbose > 1){
    std::cout << "b transformed:\n";
    printVec(b_d, K, 1);
  }
  // raw pointers
  double *tau2_ptr = thrust::raw_pointer_cast(chain.tau2.data());
  double *a_ptr = thrust::raw_pointer_cast(a_d.data());
  double *b_ptr = thrust::raw_pointer_cast(b_d.data());
  
  //generate
  getGamma<<<K, 1>>>(states, a_ptr, b_ptr, tau2_ptr);
  if(verbose > 1){
    std::cout <<"tau2 immediately after getGamma:\n";
    printVec(chain.tau2, K, 1);
  }
}

void draw_pi(curandState *states, chain_t &chain, priors_t &priors, summary2 &summary, int verbose = 0){
  int K = priors.K;
  fvec_d Tk(K);
  fvec_d Mkp1(K);
  fvec_d Vk(K, 1.0);
  if(verbose > 1){
    std::cout << "Tk init:\n";
    printVec(Tk, K, 1);
  }
  thrust::exclusive_scan(summary.Mk.rbegin(), summary.Mk.rend(), Tk.rbegin());
  if(verbose > 1){
    std::cout << "Tk filled:\n";
    printVec(Tk, K, 1);
  }
  thrust::transform(Tk.begin(), Tk.end(), Tk.begin(), thrust::placeholders::_1 + priors.alpha);
  if(verbose > 1){
    std::cout <<"Tk transformed";
    printVec(Tk, K, 1);
  }
  thrust::transform(summary.Mk.begin(), summary.Mk.end(), Mkp1.begin(), thrust::placeholders::_1 + 1.0);
  getBeta<<<K-1, 1>>>(states, thrust::raw_pointer_cast(Mkp1.data()),
                    thrust::raw_pointer_cast(Tk.data()),
                    thrust::raw_pointer_cast(Vk.data()));
  if(verbose > 1){
    std::cout <<"Vk:\n";
    printVec(Vk, K, 1);
  }
  fvec_d Ck(K, 0.0);
  transform_inclusive_scan(Vk.begin(), Vk.end()-1, Ck.begin()+1, log_1m(), thrust::plus<double>());
  if(verbose > 1){
    std::cout << "Ck:\n";
    printVec(Ck, K, 1);
  }
  transform(Vk.begin(), Vk.end(), Ck.begin(), chain.pi.begin(), exp_log_plus());
  if(verbose > 1){
    std::cout << "pi:\n";
    printVec(chain.pi, K, 1);
  }
}

void draw_zeta(curandState *states, data_t &data, chain_t &chain, priors_t &priors, int verbose=0){
  fvec_d grid(data.G*priors.K);
  cluster_weights(grid, data, chain);
  if(verbose > 1){
    std::cout << "grid:\n";
    printVec(grid, priors.K, data.G);
  }
  gnl_multinomial(chain.zeta, grid, states, priors.K, data.G);
  if(verbose > 1){
    std::cout << "(inside draw_zeta) zeta:\n";
    printVec(chain.zeta, data.G, 1);
  }
}

void draw_beta(curandState *states, data_t &data, chain_t &chain, priors_t &priors, summary2 &smry, int verbose=0){
  fvec_d prec(priors.K * data.V * data.V);
  fvec_d betahat(priors.K * data.V, 0.0);
  
  //get cluster (inv)scales
  construct_prec(prec, data, priors, chain, smry.Mk, --verbose);
  realIter prec_begin = prec.begin();
  realIter prec_end = prec.end();
  chol_multiple(prec_begin, prec_end, data.V, priors.K);
  
  //init betahat with tau2[k] * xty_sum[k] + lambda2 * mu0
  construct_weighted_sum(betahat, smry, priors, chain, --verbose);
  
  beta_hat(prec, betahat, priors.K, data.V);
  draw_MVNormal(states, betahat, prec, chain.beta, priors, --verbose);
}

void draw_pi_SD(curandState *states, chain_t &chain, priors_t &priors, summary2 &summary, int verbose = 0){
  int K = priors.K;
  fvec_d a(K);
  fvec_d b(K, 1.0);
  double *a_ptr = thrust::raw_pointer_cast(a.data());
  double *b_ptr = thrust::raw_pointer_cast(b.data());
  double *raw_ptr = thrust::raw_pointer_cast(chain.pi.data());
  getGamma<<<K, 1>>>(states, a_ptr, b_ptr, raw_ptr);
  double sum = thrust::reduce(chain.pi.begin(), chain.pi.end());
  thrust::transform(chain.pi.begin(), chain.pi.end(), chain.pi.begin(), thrust::placeholders::_1 / sum);
}

