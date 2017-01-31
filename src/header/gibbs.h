#ifndef GIBBS_H
#define GIBBS_H

#include "iter_getter.h"
#include "summary2.h"
#include "chain.h"
#include "distribution.h"
#include <thrust/iterator/reverse_iterator.h>

void draw_MVNormal(curandState *states, fvec_d &beta_hat, fvec_d &chol_prec, fvec_d &beta, priors_t &priors, summary2 &smry);

void draw_tau2(curandState *states, chain_t &chain, priors_t &priors, data_t &data, summary2 &summary);

void draw_pi(curandState *states, chain_t &chain, priors_t &priors, summary2 &summary);

#endif