#ifndef CHAIN_H
#define CHAIN_H

#include "iter_getter.h"
#include "transpose.h"

struct data_t{
  
  fvec_d yty;
  fvec_d xty;
  fvec_d ytx;
  fvec_d xtx;
  //dims
  int G;
  int V;
  int N;
  
  data_t(double* _yty, double* _xty, double* _xtx, int _G, int _V, int _N);
};

struct priors_t{
  
  int K;
  int V;
  fvec_d mu0;
  fvec_d lambda2;
  double alpha;
  double a;
  double b;
  
  priors_t(int _K, int _V, double* _mu0, double* _lambda2, double _alpha, double _a, double _b) : K(_K), V(_V), alpha(_alpha), a(_a), b(_b){
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
  fvec_d C;
  //summaries
  fvec_d probs;
  fvec_d means;
  fvec_d meansquares;
  
  chain_t(int _G, int _V, int _K, int _P, double *_beta, double *_pi, double *_tau2,
          int *_zeta, double *_C, double *_probs, double *_means, double *_meansquares):
    G(_G), V(_V), K(_K), P(_P), beta(_beta, _beta + _V*_K), pi(_pi, _pi + _K), 
    tau2(_tau2, _tau2 + _K), zeta(_zeta, _zeta + _G), C(_C, _C + _P*_V), probs(_probs, _probs + _G),
    means(_means, _means + _G*_V), meansquares(_meansquares, _meansquares + _G*_V){}
  
  void update_means(int step);
  void update_probabilities(int step);
};
  
  
struct samples_t{
  int n_iter;
  int step;
  int K_save;
  int V;
  ivec_d save_idx;
  fvec_h save_beta;
  fvec_h save_tau2;
  fvec_h save_pi;
  SCIntIter beta_iter;
  
  samples_t(int _n_iter, int _K_save, int _V, int *idx);
  void write_samples(chain_t &chain);
  
};

struct mcmc_t{

  data_t* data;
  priors_t* priors;
  chain_t* chain;
  
  mcmc_t(data_t* _data, priors_t* _priors, chain_t* _chain): data(_data), priors(_priors), chain(_chain){}
  
};
  
#endif // CHAIN_H
