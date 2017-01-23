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

prior_t Rprior_wrap(SEXP Rprior){
  int K = INTEGER(VECTOR_ELT(Rprior, 0)),
      V = INTEGER(VECTOR_ELT(Rprior, 1));
  double *mu0 = REAL(VECTOR_ELT(Rprior, 2)),
         *lambda = REAL(VECTOR_ELT(Rprior, 3)),
         alpha = REAL(VECTOR_ELT(Rprior, 4)),
         a = REAL(VECTOR_ELT(Rprior, 5)),
         b = REAL(VECTOR_ELT(Rprior, 6));
  prior_t prior(K, V, mu0, lambda, alpha, a, b);
}
