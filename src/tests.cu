#include "header/chain.h"
#include "header/iterator.h"
#include "header/printing.h"
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>


extern "C" SEXP Rdata_init(SEXP ytyR, SEXP xtyR, SEXP xtxR, SEXP G, SEXP V, SEXP N){
  int g = INTEGER(G)[0], v = INTEGER(V)[0], k = INTEGER(K)[0];
  int *zp = INTEGER(zeta);
  fvec_h xtx(v*v, 1.0);
  double *ytyp = REAL(ytyR);
  double *xtyp = REAL(xtyR);
  double *xtxp = &(xtx[0]);
  data_t data(ytyp, xtyp, ytxp, xtxp, g, v, 1);
  printVec(data.xtx, V, V);
  printVec(data.xty, V, K);
  printVec(data.ytx, K, V);
}

