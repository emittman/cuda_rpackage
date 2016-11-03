#include "summary.h"
#include "summary_fn.h"
#include "thrust.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

extern "C" SEXP summary_stats(SEXP mat, SEXP key, SEXP n_clust, SEXP verbose){
  int n_row = nrows(mat),
  n_col = ncols(mat),
  n_clustC = INTEGER(n_clust)[0],
  verboseC = INTEGER(verbose)[0];
  ivec_d hkey(INTEGER(key), INTEGER(key) + n_row);
  ivec_d hmat(INTEGER(mat), INTEGER(mat) + n_row*n_col);
  summary hsumm(n_row, n_clustC, n_col);
  thrust::copy(hmat.begin(), hmat.end(), hsumm.all.begin());
  
  hsumm.update(hkey, verboseC);
  
  SEXP clust_sums = PROTECT(allocVector(INTSXP, n_clustC*n_col));
  SEXP clust_occ  = PROTECT(allocVector(INTSXP, n_clustC));
  
  for(int i=0; i<n_clustC; ++i){
    INTEGER(clust_occ)[i] = hsumm.occupancy_count[i];
    for(int j=0; j<n_col; ++j){
      INTEGER(clust_sums)[j*n_clustC + i] = hsumm.clust_sums[j*n_clustC + i];
    }
  }
  SEXP toreturn = PROTECT(allocVector(VECSXP, 2));
  SET_VECTOR_ELT(toreturn, 0, clust_occ);
  SET_VECTOR_ELT(toreturn, 1, clust_sums);
  UNPROTECT(3);
  return toreturn;
}

extern "C" SEXP Rmy_reduce(SEXP Rvec){
  int len = length(Rvec);
  double *vecp = REAL(Rvec);
  fvec_h hvec(vecp, vecp + len);
  SEXP result = PROTECT(allocVector(REALSXP, 1));
  REAL(result)[0] = my_reduce(hvec);
  UNPROTECT(1);
  return result;
}
