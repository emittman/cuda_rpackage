#include "../header/gibbs.h"
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/reduce.h>

struct log_1m_exp {
  __host__ __device__ double operator()(double &x){
    double u = 1. - exp(x);
    if(u == 1.)
      return u-1;
    return log(u);
  }
};

struct exp_log_plus {
  __host__ __device__ double operator()(double &x, double &y){
    return exp(log(x) + y);
  }
};

void draw_MVNormal(curandState *states, fvec_d &beta_hat, fvec_d &chol_prec, fvec_d &beta, priors_t &priors, int verbose = 0){
  //no longer should be passing summary2!
  int K = priors.K;
  int V = priors.V;
  //replace current beta with standard normal draws
  int blocksize = (512 / (32*V)) * 32*V;
  int num_blocks = K*V / blocksize + 1;
  getNormal<<<num_blocks, blocksize>>>(states, K*V, thrust::raw_pointer_cast(beta.data()));
  
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

struct modify_gamma_par {
  double N;
  modify_gamma_par(double _N): N(_N){}
  template<typename T>
  __host__ __device__ void operator()(T tup){
    thrust::get<0>(tup) = thrust::get<0>(tup) + 1.0/ 2.0 * thrust::get<1>(tup) * N;
  }
};

void draw_tau2(curandState *states, chain_t &chain, priors_t &priors, data_t &data, summary2 &smry, int verbose=0){
  fvec_d sse(smry.num_occupied);
  int K = chain.K;
  fvec_d a_d(K, priors.a);
  fvec_d b_d(K, priors.b);
  if(verbose > 2){
    std::cout << "a_d filled:\n";
    printVec(a_d, K, 1);
    std::cout << "b filled:\n";
    printVec(b_d, K, 1);
  }
  
  // get SSE by cluster (for occupied clusters)
  smry.sumSqErr(sse, chain.beta, --verbose);
  if(verbose > 2){
    std::cout << "sse:\n";
    printVec(sse, smry.num_occupied, 1);
  }
  
  /*** modify gamma parameters for occupied clusters
  *
  ***/
  //a --> a + M * N/2
  modify_gamma_par f1(data.N);
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(a_d.begin(), smry.Mk.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(a_d.begin(), smry.Mk.begin())) + K,
                   f1);
  /*                 
  *typedef thrust::tuple<realIter, intIter> tuple1;
  *typedef thrust::zip_iterator<tuple1> zip1;
  *tuple1 tup1 = thrust::tuple<realIter, intIter>(a_d.begin(), smry.Mk.begin());
  *zip1 zp1 = thrust::zip_iterator<tuple1>(tup1);thrust::for_each(zp1, zp1 + K, f1);
  */
  
  if(verbose > 1){
    std::cout << "a transformed:\n";
    printVec(a_d, K, 1);
  }
  
  // b --> b + sse/2
  modify_gamma_par f2(1.0);
  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator<realIter, intIter>(b_d.begin(), smry.occupied.begin()), sse.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator<realIter, intIter>(b_d.begin(), smry.occupied.begin()), sse.begin())) + smry.num_occupied,
    f2
  )
  /*
  *typedef thrust::permutation_iterator<realIter, intIter> FltPermIter;
  *FltPermIter b_occ = thrust::permutation_iterator<realIter, intIter>(b_d.begin(), smry.occupied.begin());
  *typedef thrust::tuple<FltPermIter, realIter> tuple2;
  *typedef thrust::zip_iterator<tuple2> zip2;
  *tuple2 tup2 = thrust::tuple<FltPermIter, realIter>(b_occ, sse.begin());
  *zip2 zp2 = thrust::zip_iterator<tuple2>(tup2);
  *thrust::for_each(zp2, zp2 + smry.num_occupied, f2);
  */
  if(verbose > 1){
    std::cout << "b transformed:\n";
    printVec(b_d, K, 1);
  }
  
  // raw pointers
  double *tau2_ptr = thrust::raw_pointer_cast(chain.tau2.data());
  double *a_ptr = thrust::raw_pointer_cast(a_d.data());
  double *b_ptr = thrust::raw_pointer_cast(b_d.data());
  
  //generate
  int blocksize = 512;
  int num_blocks = K/blocksize + 1;
  getGamma<<<num_blocks, blocksize>>>(states, K, a_ptr, b_ptr, tau2_ptr, false);
  if(verbose > 1){
    std::cout <<"tau2 immediately after getGamma:\n";
    printVec(chain.tau2, K, 1);
  }
}

void draw_pi(curandState *states, chain_t &chain, priors_t &priors, summary2 &summary, int verbose = 0){
  int K = priors.K;
  fvec_d Tk(K);
  fvec_d Mkp1(K);
  fvec_d Vk(K, 0.0);
  if(verbose > 1){
    std::cout << "Tk init:\n";
    printVec(Tk, K, 1);
  }
  thrust::exclusive_scan(summary.Mk.rbegin(), summary.Mk.rend(), Tk.rbegin());
  if(verbose > 1){
    std::cout << "Tk filled:\n";
    printVec(Tk, K, 1);
  }
  thrust::transform(Tk.begin(), Tk.end(), Tk.begin(), thrust::placeholders::_1 + chain.alpha);
  if(verbose > 1){
    std::cout <<"Tk transformed";
    printVec(Tk, K, 1);
  }
  thrust::transform(summary.Mk.begin(), summary.Mk.end(), Mkp1.begin(), thrust::placeholders::_1 + 1.0);
  
  int blocksize = 512;
  int num_blocks = (K-1)/blocksize + 1;
  getBeta<<<num_blocks, blocksize>>>(states, K-1, thrust::raw_pointer_cast(Mkp1.data()),
                    thrust::raw_pointer_cast(Tk.data()),
                    thrust::raw_pointer_cast(Vk.data()),
                    true);
  if(verbose > 1){
    std::cout <<"Vk:\n";
    printVec(Vk, K, 1);
  }
  
  fvec_d Ck(K, 0.0);
  
  transform_inclusive_scan(Vk.begin(), Vk.end()-1, Ck.begin()+1, log_1m_exp(), thrust::plus<double>());
  
  if(verbose > 1){
    std::cout << "Ck:\n";
    printVec(Ck, K, 1);
  }
  transform(Vk.begin(), Vk.end(), Ck.begin(), chain.pi.begin(), thrust::plus<double>());
  if(verbose > 1){
    std::cout << "log pi:\n";
    printVec(chain.pi, K, 1);
  }
}

void draw_zeta(curandState *states, data_t &data, chain_t &chain, priors_t &priors, int verbose=0){
  if(verbose>1){
    std::cout << "Sampling zeta...\n";
  }
  fvec_d grid(data.G*priors.K);
  if(verbose>1){
    std::cout << "Computing weights...\n";
  }
  
  if(chain.voom){
    cluster_weights_voom(grid, data, chain, verbose);
  } else{
    cluster_weights_no_voom(grid, data, chain, verbose);
  }
  
  if(verbose > 2){
    std::cout << "grid:\n";
    printVec(grid, priors.K, data.G);
  }
  if(verbose>1){
    std::cout << "Mutinomial samping...\n";
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
  construct_prec(prec, smry, priors, chain, --verbose);
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
  thrust::constant_iterator<double> adk = thrust::constant_iterator<double>(chain.alpha/K);
  thrust::transform(adk, adk+K, summary.Mk.begin(), a.begin(), thrust::plus<double>());
  if(verbose > 0){
    printVec(a, K, 1);
  }
  double *a_ptr = thrust::raw_pointer_cast(a.data());
  double *b_ptr = thrust::raw_pointer_cast(b.data());
  double *raw_ptr = thrust::raw_pointer_cast(chain.pi.data());
  
  int blocksize = 512;
  int num_blocks = K/blocksize + 1;
  getGamma<<<num_blocks, blocksize>>>(states, K, a_ptr, b_ptr, raw_ptr, true);
  if(verbose > 0){
    std::cout << "log pi before normalization:\n";
    printVec(chain.pi, K, 1);
  }
  realIter pi_begin = chain.pi.begin();
  realIter pi_end = chain.pi.end();
  
  log_sum_exp lse_fnc;
  double init = chain.pi[0];
  double log_sum = thrust::reduce(pi_begin+1, pi_end, init, lse_fnc);
  if(verbose > 0){
    std::cout << "normalization constant: " << exp(log_sum) << std::endl;
  }
  thrust::transform(pi_begin, pi_end, pi_begin, thrust::placeholders::_1 - log_sum);
  
  if(verbose > 0){
    std::cout << "log pi after normalization: \n";
    printVec(chain.pi, K, 1);
  }
}

void draw_alpha(chain_t &chain, priors_t &priors, int verbose){
  int K = priors.K;
  GetRNGstate();
  double Aupdate = priors.A + K - 1;
  double Bupdate = priors.B - chain.pi[K-1];
  double draw = Rf_rgamma(priors.A + K - 1, 1/(priors.B - chain.pi[K-1]));
  PutRNGstate();
  chain.alpha = draw;
  if(verbose>0){
    std::cout << "Updated A: " << Aupdate << ", updated B: " << Bupdate << ", draw = " << draw << std::endl;
  }
}

/*
void draw_alpha_SD(chain_t &chain, priors_t &priors, int verbose, bool adapt){
  int K = priors.K;
  double prev = chain.alpha;
  double logprob, logu;
  GetRNGstate();
  double ppsl = Rf_rnorm(prev, chain.s_RW_alpha);
  if(verbose > 1){
    std::cout << "previous value for alpha: " << prev;
    std::cout << ",\t proposal: " << ppsl;
  }
  
  if(ppsl>0){
    double mean_logpi = thrust::reduce(chain.pi.begin(), chain.pi.end(), 0, thrust::plus<double>())/K;
    logprob = Rf_lgammafn(ppsl) - Rf_lgammafn(prev) - K * (Rf_lgammafn(ppsl/K) - Rf_lgammafn(prev/K));
    logprob += (priors.A-1)*(log(ppsl) - log(prev)) + (mean_logpi - priors.B) * (ppsl - prev);
    logu = log(Rf_runif(0, 1));
    if(logu < logprob){
      chain.alpha = ppsl;
    }
    if(verbose > 1){
      std::cout << ",\t acceptance prob.: " << exp(logprob) << std::endl;
    }
  } else{
    if(verbose > 1){
      std::cout << ",\t acceptance prob.: ZERO" << std::endl;
    }
  }
  if(adapt){
    if(ppsl<0 || logu > logprob){
      chain.s_RW_alpha *= .95;
      if(verbose>0){
        std::cout << "Proposal rejected: new sd = " << chain.s_RW_alpha << std::endl;
      }
    } else {
      chain.s_RW_alpha /= .95;
      if(verbose>0){
        std::cout << "Proposal accepted: new sd = " << chain.s_RW_alpha << std::endl;
      }
    }
  }
  PutRNGstate();
}
*/

double target_alpha::operator()(double arg){
  if(arg<0) return -INFINITY;
  else return lgamma(arg) - K*lgamma(arg/K) + (A-1)*log(arg) + (mean_logpi - B)*arg;
}

    
void draw_alpha_SD_slice(chain_t &chain, priors_t &priors, int verbose, bool adapt, int warmup_iter){
  double mean_logpi = thrust::reduce(chain.pi.begin(), chain.pi.end(), 0, thrust::plus<double>()) / priors.K;
  if(verbose>0){
    std::cout << "sum(log(pi))/K = " << mean_logpi <<"\n";
  }
  target_alpha f(priors.A, priors.B, priors.K, mean_logpi);

  double y, U, L, R, V, x0 = chain.alpha, w0 = chain.slice_width;
  int J, K;

  if(verbose>0){
    std::cout << "chain.alpha=" << x0 << "\n";
  }
  if(verbose>0){
    std::cout << "chain.slice_width" << w0 << "\n";
  }

  y = f(x0) + log(Rf_runif(0,1));
  U = Rf_runif(0,1);
  L = x0 - w0 * U;
  R = L + w0;
  V = Rf_runif(0,1);
  J = floor(chain.max_steps * V);
  K = (chain.max_steps - 1) - J;
  while(J>0 & y<f(L)){
    L = L - w0;
    J = J - 1;
  }
  while(K>0 & y<f(R)){
    R = R + w0;
    K = K - 1;
  }
  
  double x;
  do {
    x = Rf_runif(L, R);
    if (x > x0)
      R = x;
    else
      L = x;
  } while(f(x) < y);
  chain.alpha = x;
  if(adapt){
    double w = 2*fabs(x-x0);
    if(chain.slice_width > w){
      chain.slice_width *= .99;
    } else{
      chain.slice_width *= 1.01;
    }
  }
}

