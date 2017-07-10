#include "header/chain.h"
#include "header/iterator.h"
#include "header/printing.h"
#include "header/wrap_R.h"
#include <R.h>
#include <Rinternals.h>
#include "header/quadform2.h"

extern "C" SEXP Rquadform_multipleK(SEXP Rbeta, SEXP Rxtx, SEXP Kv, SEXP Vv){

  double *betaptr = REAL(Rbeta), *xtxptr = REAL(Rxtx);
  int K = INTEGER(Kv)[0], V = INTEGER(Vv)[0];

  fvec_d beta(betaptr, betaptr+K*V);
  fvec_d xtx(xtxptr, xtxptr+V*V);
  fvec_d result(K);
  
  quadform_multipleK(beta, xtx, result, K, V);
  
  SEXP out = PROTECT(allocVector(REALSXP, K));
  double *outp = REAL(out);
  thrust::copy(result.begin(), result.end(), outp);
  
  return out;
}

extern "C" SEXP Rquadform_multipleMatch(SEXP Rbeta, SEXP Rxtx, SEXP Gv, SEXP Kv, SEXP Vv){

  double *betaptr = REAL(Rbeta), *xtxptr = REAL(Rxtx);
  int G = INTEGER(Gv)[0], K = INTEGER(Kv)[0], V = INTEGER(Vv)[0];

  fvec_d beta(betaptr, betaptr+K*V);
  fvec_d xtx(xtxptr, xtxptr+V*V*K);
  fvec_d result(K);
  
  quadform_multipleMatch(beta, xtx, result, K, V);
  
  SEXP out = PROTECT(allocVector(REALSXP, K));
  double *outp = REAL(out);
  thrust::copy(result.begin(), result.end(), outp);
  
  return out;
}

extern "C" SEXP Rquadform_multipleGK(SEXP Rbeta, SEXP Rxtx, SEXP Gv, SEXP Kv, SEXP Vv){

  double *betaptr = REAL(Rbeta), *xtxptr = REAL(Rxtx);
  int G = INTEGER(Gv)[0], K = INTEGER(Kv)[0], V = INTEGER(Vv)[0];

  fvec_d beta(betaptr, betaptr+K*V);
  fvec_d xtx(xtxptr, xtxptr+V*V*K);
  fvec_d result(G*K);
  
  quadform_multipleGK(beta, xtx, result, G, K, V);
  
  SEXP out = PROTECT(allocVector(REALSXP, G*K));
  double *outp = REAL(out);
  thrust::copy(result.begin(), result.end(), outp);
  
  return out;
}

