#include "header/cuda_usage.h"
#include "header/chain.h"
#include "header/iterator.h"
#include "header/printing.h"
#include "header/cluster_probability.h"
#include "header/multinomial.h"
#include "header/distribution.h"
#include "header/beta_hat.h"
#include "header/wrap_R.h"
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
  int g = INTEGER(G)[0], v = INTEGER(V)[0], n = INTEGER(N)[0], k = INTEGER(K)[0];
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

extern "C" SEXP Rnormalize_wts(SEXP grid, SEXP dim1, SEXP dim2){
  int k=INTEGER(dim1)[0], g = INTEGER(dim2)[0];
  fvec_h grd_h(REAL(grid), REAL(grid) + k*g);
  fvec_d grd_d(k*g);
  thrust::copy(grd_h.begin(), grd_h.end(), grd_d.begin());
  normalize_wts(grd_d, k, g);
  thrust::copy(grd_d.begin(), grd_d.end(), grd_h.begin());
  SEXP out = PROTECT(allocVector(REALSXP, g*k));
  for(int i=0; i<k*g; ++i) REAL(out)[i] = grd_h[i];
  UNPROTECT(1);
  return out;
}


extern "C" SEXP RgetUniform(SEXP Rseed, SEXP upperR){

  int n = length(upperR), seed = INTEGER(Rseed)[0];

  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, n * sizeof(curandState)));
  
  //temporary memory
  fvec_h upper(REAL(upperR), REAL(upperR) + n);
  fvec_d upper_d(upper.begin(), upper.end());
  
  double *upper_d_ptr = thrust::raw_pointer_cast(upper_d.data());
    
  //set up RNGs
  setup_kernel<<<n,1>>>(seed, devStates);
  
  //sample from U(0, upper)
  getUniform<<<n,1>>>(devStates, upper_d_ptr);
 
  thrust::copy(upper_d.begin(), upper_d.end(), upper.begin());
  
  SEXP out = PROTECT(allocVector(REALSXP, n));
  for(int i = 0; i < n; ++i) REAL(out)[i] = upper_d[i];
  
  UNPROTECT(1);
  return out;
}

extern "C" SEXP Rgnl_multinomial(SEXP Rseed, SEXP probs, SEXP K, SEXP G){
  int k = INTEGER(K)[0], g = INTEGER(G)[0], seed = INTEGER(Rseed)[0];

  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, g * sizeof(curandState)));
  
  //temporary memory
  fvec_h probs_h(REAL(probs), REAL(probs) + g*k);
  fvec_d probs_d(probs_h.begin(), probs_h.end());
  ivec_h zeta_h(g);
  ivec_d zeta_d(g);
  
  double *probs_d_ptr = thrust::raw_pointer_cast(probs_d.data());
  
  //set up RNGs
  setup_kernel<<<g,1>>>(seed, devStates);
  
  //get multinomial draws
  gnl_multinomial(zeta_d, probs_d, devStates, k, g);
 
  thrust::copy(probs_d.begin(), probs_d.end(), probs_h.begin());
  thrust::copy(zeta_d.begin(), zeta_d.end(), zeta_h.begin());
  
  SEXP out = PROTECT(allocVector(VECSXP, 2));
  SEXP out_z = PROTECT(allocVector(INTSXP, g));
  SEXP out_p = PROTECT(allocVector(REALSXP, k*g));
  
  for(int i = 0; i < g; ++i){
    INTEGER(out_z)[i] = zeta_h[i];
    for(int j=0; j < k; ++j){
      REAL(out_p)[i*k + j] = probs_h[i*k + j];
    }
  }
  
  SET_VECTOR_ELT(out, 0, out_z);
  SET_VECTOR_ELT(out, 1, out_p);
  
  UNPROTECT(3);
  return out;
}

extern "C" SEXP Rbeta_hat(SEXP R_Lvec, SEXP R_xty, SEXP K, SEXP V){
  int k=INTEGER(K)[0], v=INTEGER(V)[0];
  fvec_h L_h(REAL(R_Lvec), REAL(R_Lvec)+k*v*v), b_h(REAL(R_xty), REAL(R_xty)+k*v);
  fvec_d L_d(k*v*v);
  fvec_d b_d(k*v);
  thrust::copy(L_h.begin(), L_h.end(), L_d.begin());
  thrust::copy(b_h.begin(), b_h.end(), b_d.begin());
  beta_hat(L_d, b_d, k, v);
  thrust::copy(b_d.begin(), b_d.end(), b_h.begin());
  SEXP out = PROTECT(allocVector(REALSXP, k*v));
  for(int i=0; i<k*v; ++i) REAL(out)[i] = b_h[i];
  UNPROTECT(1);
  return out;
}


extern "C" SEXP Rtest_data_wrap(SEXP Rdata){
  data_t data = Rdata_wrap(Rdata);
  printVec(data.ytx, data.G, data.V);
  SEXP out = PROTECT(allocVector(INTSXP, 1));
  INTEGER(out)[0] = 0;
  UNPROTECT(1);
  return out;
}