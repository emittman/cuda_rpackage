#include "header/chain.h"
#include "header/iterator.h"
#include "header/printing.h"
#include "header/cluster_probability.h"
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>


extern "C" SEXP Rdata_init(SEXP ytyR, SEXP xtyR, SEXP xtxR, SEXP G, SEXP V, SEXP N){
  int g = INTEGER(G)[0], v = INTEGER(V)[0], n = INTEGER(N)[0];
  fvec_h xtx(v*v, 1.0);
  double *ytyp = REAL(ytyR);
  double *xtyp = REAL(xtyR);
  double *xtxp = &(xtx[0]);
  data_t data(ytyp, xtyp, xtxp, g, v, n);
  printVec(data.xtx, v, v);
  printVec(data.xty, v, g);
  printVec(data.ytx, g, v);
  SEXP zero = PROTECT(allocVector(INTSXP, 1));
  INTEGER(zero)[0] = 0;
  UNPROTECT(1);
  return zero;
}

extern "C" SEXP Rcluster_weights(SEXP A, SEXP B, SEXP C, SEXP D, SEXP E, SEXP G, SEXP V, SEXP N, SEXP K){
  int g = INTEGER(G)[0], v = INTEGER(v)[0], n = INTEGER(N)[0], k = INTEGER(K)[0];
  fvec_h a_h(REAL(A), REAL(A) + g*k);
  fvec_d a(g*k);
  thrust::copy(a_h.begin(), a_h.end(), a.begin());
  fvec_d b(REAL(B), REAL(B) + k);
  fvec_d c(REAL(C), REAL(C) + k);
  fvec_d d(REAL(D), REAL(D) + g);
  fvec_d e(REAL(E), REAL(E) + k);
  cluster_weights(a, b, c, d, e, g, v, n, k);
  thrust::copy(a.begin(), a.end(), a_h.begin());
  SEXP OUT = PROTECT(allocVector(REALSXP, g*k));
  for(int i=0; i<g*k; ++i) REAL(OUT)[i] = a_h[i];
  UNPROTECT(1);
  return OUT;
}


