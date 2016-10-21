#include <thrust/device_vector.h>
#include "iterator2.h"
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

extern "C" SEXP Rquad_form_multi(SEXP A, SEXP x, SEXP n, SEXP dim){
  
  double *Aptr = REAL(A), *xptr = REAL(x);
  int N = INTEGER(n)[0], D = INTEGER(n)[0];
  
  thrust::device_vector<double> dA(Aptr, Aptr+D*D);
  thrust::device_vector<double> dx(xptr, xptr+N*D);
  thrust::device_vector<double> dy(N);
  
  quad_form_multi(dA, dx, dy, N, D);
  
  SEXP y = PROTECT(allocVector(REALSXP, N));
  for(int i=0; i<N; ++i)
    REAL(y)[i] = dy[i];
  
  UNPROTECT(1);
  return y;
}