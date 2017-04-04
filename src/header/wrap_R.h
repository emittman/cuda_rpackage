#ifndef WRAP_R
#define WRAP_R

#include "chain.h"
#include <Rinternals.h>

data_t Rdata_wrap(SEXP Rdata, int verbose=0);
priors_t Rpriors_wrap(SEXP Rpriors, int verbose=0);
chain_t Rchain_wrap(SEXP Rchain, int verbose=0);
SEXP Csamples_wrap(samples_t &samples, int verbose=0);
SEXP Cchain_wrap(chain_t &chain, int verbose=0);
SEXP Cstate_wrap(chain_t &chain, int verbose=0);
#endif