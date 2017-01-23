#include "../header/wrap_R.h"

data_t Rdata_wrap(SEXP Rdata){
  double *yty = REAL(VECTOR_ELT(Rdata, 0)),
    *xty = REAL(VECTOR_ELT(Rdata, 1)),
    *xtx = REAL(VECTOR_ELT(Rdata, 2));
  int G = INTEGER(VECTOR_ELT(Rdata, 3))[0],
      V = INTEGER(VECTOR_ELT(Rdata, 4))[0],
      N = INTEGER(VECTOR_ELT(Rdata, 5))[0];
  data_t data(yty, xty, xtx, G, V, N);
  return data;
}

priors_t Rpriors_wrap(SEXP Rpriors){
  int K = INTEGER(VECTOR_ELT(Rpriors, 0))[0],
      V = INTEGER(VECTOR_ELT(Rpriors, 1))[0];
  double *mu0 = REAL(VECTOR_ELT(Rpriors, 2)),
         *lambda = REAL(VECTOR_ELT(Rpriors, 3)),
         alpha = REAL(VECTOR_ELT(Rpriors, 4))[0],
         a = REAL(VECTOR_ELT(Rpriors, 5))[0],
         b = REAL(VECTOR_ELT(Rpriors, 6))[0];
  priors_t priors(K, V, mu0, lambda, alpha, a, b);
  return priors;
}
