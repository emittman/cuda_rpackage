#ifndef GIBBS_H
#define GIBBS_H

#include "iter_getter.h"
#include "summary2.h"
#include "chain.h"
#include "distribution.h"

void draw_tau2(curandState *states, chain_t &chain, prior_t &prior, summary2 &summary);

#endif