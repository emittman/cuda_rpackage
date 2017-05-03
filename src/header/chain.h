#ifndef CHAIN_H
#define CHAIN_H

#include "iter_getter.h"
#include "transpose.h"
#include "printing.h"

//forward declaration
struct summary2;

struct data_t{
  
  fvec_d yty;
  fvec_d xty;
  fvec_d ytx;
  fvec_d xtx;
  fvec_d txtx;
  
  //dims
  int G;
  int V;
  int N;
  
  //indicates whether to use precision weights
  bool voom;
  
  data_t(double* _yty, double* _xty, double* _xtx, int _G, int _V, int _N, bool _voom);
};

struct priors_t{
  
  int K;
  int V;
  fvec_d mu0;
  fvec_d lambda2;
  double a;
  double b;
  double A;
  double B;
  
  priors_t(int _K, int _V, double* _mu0, double* _lambda2, double _a, double _b, double _A, double _B) :
    K(_K), V(_V), a(_a), b(_b), A(_A), B(_B){
    mu0 = fvec_d(_mu0, _mu0 + V);
    lambda2 = fvec_d(_lambda2, _lambda2 + V);
  }

};

struct chain_t{
  
  //dims
  int G;
  int V;
  int K;
  int n_hyp;
  int P;
  //parameters
  fvec_d beta;
  fvec_d pi;
  fvec_d tau2;
  ivec_d zeta;
  double alpha;
  //contrasts
  ivec_d C_rowid;
  fvec_d C;
  //summaries
  fvec_d probs;
  fvec_d means;
  fvec_d meansquares;
  //tuning parameter
  //double s_RW_alpha;
  double slice_width;
  int max_steps;
  
  chain_t(int _G, int _V, int _K, int _n_hyp, int *_C_rowid, int _P, double *_beta, double *_pi, double *_tau2,
          int *_zeta, double _alpha, double *_C, double *_probs, double *_means, double *_meansquares, double _slice_width,
          int _max_width):
    G(_G), V(_V), K(_K), n_hyp(_n_hyp), C_rowid(_C_rowid, _C_rowid + _P), P(_P), beta(_beta, _beta + _V*_K), pi(_pi, _pi + _K), 
    tau2(_tau2, _tau2 + _K), zeta(_zeta, _zeta + _G), alpha(_alpha), C(_C, _C + _P*_V), probs(_probs, _probs + _n_hyp*_G),
    means(_means, _means + _G*_V), meansquares(_meansquares, _meansquares + _G*_V), slice_width(_slice_width), max_width(_max_width){}
  
  void update_means(int step);
  void update_probabilities(int step);
};

//      chain_t chain(G, V, K, n_hyp, C_rowid, P, beta, pi, tau2, zeta, C, probs, means, meansquares);
 
  
struct samples_t{
  int n_save_g;
  int n_save_P;
  int step_g;
  int step_P;
  int G_save;
  int K;
  int V;
  bool alpha_fixed;
  ivec_d save_idx;
  fvec_h save_beta;
  fvec_h save_tau2;
  fvec_h save_P;
  ivec_h save_max_id;
  ivec_h save_num_occupied;
  thrust::host_vector<double> save_alpha;

  samples_t(int _n_save_g, int _n_save_P, int _G_save, int _K, int _V, int *idx, bool _alpha_fixed);
  void write_g_samples(chain_t &chain, summary2 &smry);
  void write_P_samples(chain_t &chain);
};

struct mcmc_t{

  data_t* data;
  priors_t* priors;
  chain_t* chain;
  
  mcmc_t(data_t* _data, priors_t* _priors, chain_t* _chain): data(_data), priors(_priors), chain(_chain){}
  
};
  
#endif // CHAIN_H
