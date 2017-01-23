#ifndef WRAP_R
#define WRAP_R

#include "chain.h"
#include <Rinternals.h>

data_t Rdata_wrap(SEXP Rdata);
priors_t Rpriors_wrap(SEXP Rpriors);

#endif