#ifndef WRAP_R
#define WRAP_R

#include "chain.h"
#include <Rinternals.h>

data_t Rdata_wrap(SEXP Rdata);
priors_t Rpriors_wrap(SEXP Rpriors);
chain_t Rchain_wrap(SEXP Rchain);
SEXP Csamples_wrap(samples_t &samples);
SEXP Cchain_wrap(chain_t &chain)
#endif