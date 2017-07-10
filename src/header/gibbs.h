#ifndef GIBBS_H
#define GIBBS_H

#include "iter_getter.h"
#include "summary2.h"
#include "chain.h"
#include "distribution.h"
#include "cluster_probability.h"
#include "quadform2.h"
#include "multinomial.h"
#include "beta_hat.h"
#include "construct_prec.h"
#include "cholesky.h"
#include <thrust/iterator/reverse_iterator.h>
#include <Rmath.h>
// This prevents the replacement of "beta" by Rmath.h
#ifdef beta
#undef beta
#endif

void draw_MVNormal(curandState *states, fvec_d &beta_hat, fvec_d &chol_prec, fvec_d &beta, priors_t &priors, int verbose);

void draw_tau2(curandState *states, chain_t &chain, priors_t &priors, data_t &data, summary2 &summary, int verbose);

void draw_pi(curandState *states, chain_t &chain, priors_t &priors, summary2 &summary, int verbose);

void draw_pi_SD(curandState *states, chain_t &chain, priors_t &priors, summary2 &summary, int verbose);

void draw_zeta(curandState *states, data_t &data, chain_t &chain, priors_t &priors, int verbose);

void draw_beta(curandState *states, data_t &data, chain_t &chain, priors_t &priors, summary2 &smry, int verbose);

void draw_alpha(chain_t &chain, priors_t &priors, int verbose);

//void draw_alpha_SD(chain_t &chain, priors_t &priors, int verbose, bool adapt);

void draw_alpha_SD_slice(chain_t &chain, priors_t &priors, int verbose, bool adapt, int warmup_iter);

struct target_alpha{
  double A;
  double B;
  double K;
  double mean_logpi;
  target_alpha(double _A, double _B, double _K, double _mean_logpi):
    A(_A), B(_B), K(_K), mean_logpi(_mean_logpi){}
  double operator()(double arg);
};
#endif