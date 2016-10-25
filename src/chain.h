#ifndef CHAIN_H
#define CHAIN_H

#include "iterator2.h"

struct data_t{
  
  fvec_d yty;
  fvec_d xty;
  fvec_d ytx;
  fvec_d xtx;
  //dims
  int G;
  int V;
  int N;
  
  data_t(double* _y, double* _x, int _G, int _V, int _N): yty(_G), xty(_G*_V), ytx(_G*_V), xtx(_V*_V), G(_G), V(_V), N(_N) {

    //temporaries for initial calculations
    fvec_h yty_h(G);
    fvec_h xty_h(G*V);
    fvec_h ytx_h(G*V);
    fvec_h xtx_h(V*V);
    //multiply a bunch of matrices
    
    //copy to device
    thrust::copy(yty_h.begin(), yty_h.end(), yty.begin());
    thrust::copy(xty_h.begin(), xty_h.end(), xty.begin());
    thrust::copy(ytx_h.begin(), ytx_h.end(), ytx.begin());
    thrust::copy(xtx_h.begin(), xtx_h.end(), xtx.begin());
  }
  
};

struct priors_t{
  
  int V;
  fvec_d mu0;
  fvec_d lambda2;
  double alpha;
  double a;
  double b;
  
  priors_t(int _V, double* _mu0, double* _lambda2, double _alpha, double _a, double _b) : V(_V), alpha(_alpha), a(_a), b(_b){
    mu0 = fvec_d(_mu0, _mu0 + V);
    lambda2 = fvec_d(_lambda2, _lambda2 + V);
  }

};

struct chain_t{
  
  //dims
  int G;
  int V;
  int K;
  int P;
  //parameters
  fvec_d beta;
  fvec_d pi;
  fvec_d tau2;
  ivec_d zeta;
  //contrasts
  ivec_d C;
  //summaries
  fvec_d probs;
  fvec_d means;
  fvec_d meansquares;
  
  chain_t(int* _C, int _G, int _V, int _N, int _P): G(_G), V(_V), K(_K), P(_P), beta(_K*_V), pi(_K),tau2(_K),
                                                  zeta(_K), C(_C, _C + _P*_V), probs(_G*_P), means(_G), meansquares(_G){}
  
};
  
struct mcmc_t{

  data_t* data;
  priors_t* priors;
  chain_t* chain;
  
  mcmc_t(data_t* _data, priors_t* _priors, chain_t* _chain): data(_data), priors(_priors), chain(_chain){}
  
};

  
  // void initialize_chain(chain_t &chain){
  //   // Fill with default/"agnostic" starting values
  //   for(int i=0; i<chain.K; i++){
  //     chain.pi[i] = 1.0/(double)chain.K;
  //     chain.sigma2[i] = 1.0;
  //   }
  //   
  //   for(int i=0; i<chain.K*chain.V; i++){
  //     chain.beta[i] = rnorm(0,chain.lambda2);
  //   }
  //   
  // }
  // 
  // void print_chain_state(chain_t &chain){
  //   
  //   Rprintf("z:\n");
  //   print_mat(chain.z, 1, chain.G);
  //   Rprintf("Gk:\n");
  //   print_mat(chain.Gk, 1, chain.K);
  //   Rprintf("beta:\n");
  //   print_mat(chain.beta, chain.V, chain.K);
  //   Rprintf("pi:\n");
  //   print_mat(chain.pi, 1, chain.K);
  //   Rprintf("sigma2:\n");
  //   print_mat(chain.sigma2, 1, chain.K);
  //   Rprintf("lambda2:\n %lf \n", chain.lambda2);
  //   Rprintf("alpha:\n %lf \n", chain.alpha);
  //   Rprintf("a:\n %lf \n", chain.a);
  //   Rprintf("b:\n %lf \n", chain.b);
  // }
  
#endif // CHAIN_H