#include "header/cuda_usage.h"
#include "header/cholesky.h"
#include "header/quad_form.h"
#include "header/summary2.h"
#include "header/chain.h"
#include "header/iterator.h"
#include "header/construct_prec.h"
#include "header/distribution.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

extern "C" SEXP RgetDeviceCount(){
  int count = 0;
  cudaGetDeviceCount(&count);
  
  SEXP result = PROTECT(allocVector(INTSXP, 1));
  INTEGER(result)[0] = count;
  UNPROTECT(1);
  
  return result;
}

extern "C" SEXP RsetDevice(SEXP device) {
  SEXP result = PROTECT(allocVector(INTSXP, 1));
  INTEGER(result)[0] = setDevice(INTEGER(device)[0]);
  UNPROTECT(1);
  return result;
}

extern "C" SEXP RgetDevice(){
  int device = 0;
  cudaGetDevice(&device);
  
  SEXP result = PROTECT(allocVector(INTSXP, 1));
  INTEGER(result)[0] = device;
  UNPROTECT(1);
  
  return result;
}

//wrapper to chol_multiple which does in-place cholesky decomposition on a (flattened) array of matrices
extern "C" SEXP Rchol_multiple(SEXP all, SEXP arraydim, SEXP n_array){
  int dim = INTEGER(arraydim)[0];
  int reps = INTEGER(n_array)[0];
  double *aptr = REAL(all);
  fvec_d dvec(aptr, aptr + length(all));
  realIter begin = dvec.begin();
  realIter end = dvec.end();
  chol_multiple(begin, end, dim, reps);
  fvec_h hvec(dvec.begin(), dvec.end());
  SEXP out = PROTECT(allocVector(REALSXP, length(all)));
  for(int i=0; i<length(all); ++i)
    REAL(out)[i] = hvec[i];
  UNPROTECT(1);
  return out;
}

extern "C" SEXP Rconstruct_prec(SEXP xtx, SEXP Mk, SEXP lam, SEXP tau, SEXP K, SEXP V){
  int dim = INTEGER(V)[0];
  int num_clusts = INTEGER(K)[0];
  int total_size = dim * dim * num_clusts;
  fvec_d prec(total_size);
  double *xtx_ptr = REAL(xtx);
  int *Mk_ptr = INTEGER(Mk);
  double *lam_ptr = REAL(lam);
  double *tau_ptr = REAL(tau);
  
  fvec_d dev_xtx(xtx_ptr, xtx_ptr + dim*dim);
  ivec_d dev_Mk(Mk_ptr, Mk_ptr + num_clusts);
  fvec_d dev_lam(lam_ptr, lam_ptr+dim);
  fvec_d dev_tau(tau_ptr, tau_ptr+num_clusts);
  construct_prec(prec.begin(), prec.end(), dev_lam.begin(), dev_lam.end(), dev_tau.begin(), dev_tau.end(),
                 dev_Mk.begin(), dev_Mk.end(), dev_xtx.begin(), dev_xtx.end(), num_clusts, dim);
  
  SEXP out_prec = PROTECT(allocVector(REALSXP, total_size));
  for(int i=0; i<total_size; ++i)
    REAL(out_prec)[i] = prec[i];
  UNPROTECT(1);
  return out_prec;
}

extern "C" SEXP Rbeta_rng(SEXP a, SEXP b){

  int n = length(a);

  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, n * sizeof(curandState)));
  
  //temporary memory
  
  fvec_d out(n);
 
  double *aptr = REAL(a);
  double *bptr = REAL(b);
  fvec_d da(aptr, aptr+n);
  fvec_d db(bptr, bptr+n);
  
  double *outptr = thrust::raw_pointer_cast(out.data());
  double *daptr = thrust::raw_pointer_cast(da.data());
  double *dbptr = thrust::raw_pointer_cast(db.data());
    
  //set up RNGs
  setup_kernel<<<n,1>>>(devStates);
  
  //sample from Beta(a, b)
  getBeta<<<n,1>>>(devStates, daptr, dbptr, outptr);
  
  //transfer memory
  SEXP Rout = PROTECT(allocVector(REALSXP, n));
  for(int i=0; i<n; ++i)
    REAL(Rout)[i] = out[i];
  
  //clean up
  CUDA_CALL(cudaFree(devStates));
  UNPROTECT(1);
  
  return Rout;
}

extern "C" SEXP Rquad_form_multi(SEXP A, SEXP x, SEXP n, SEXP dim){

  double *Aptr = REAL(A), *xptr = REAL(x);
  int N = INTEGER(n)[0], D = INTEGER(dim)[0];

  fvec_d dA(Aptr, Aptr+D*D);
  fvec_d dx(xptr, xptr+N*D);
  fvec_d dy(N);

  quad_form_multi(dA, dx, dy, N, D);

  SEXP y = PROTECT(allocVector(REALSXP, N));
  for(int i=0; i<N; ++i)
    REAL(y)[i] = dy[i];

  UNPROTECT(1);
  return y;
}

extern"C" SEXP Rsummary2(SEXP zeta, SEXP ytyR, SEXP ytxR, SEXP xtyR, SEXP G, SEXP V, SEXP K){
  int g = INTEGER(G)[0], v = INTEGER(V)[0], k = INTEGER(K)[0];
  int *zp = INTEGER(zeta);
  fvec_h yty(REAL(ytyR), REAL(ytyR) + g);
  fvec_h ytx(REAL(ytxR), REAL(ytxR) + g*v);
  fvec_h xty(REAL(xtyR), REAL(xtyR) + g*v);
  fvec_h xtx(v*v, 1.0);
  double *ytyp = &(yty[0]);
  double *ytxp = &(ytx[0]);
  double *xtyp = &(xty[0]);
  double *xtxp = &(xtx[0]);
  data_t data(ytyp, xtyp, ytxp, xtxp, g, v, 1);
  
  ivec_d ZETA(zp, zp+g);
  summary2 smry(g, k, v, ZETA, data);
  
  smry.print_Mk();
  smry.print_yty();
  smry.print_xty();
  
  SEXP out = PROTECT(allocVector(INTSXP, 1));
  INTEGER(out)[0] = smry.num_occupied;
  
  UNPROTECT(1);
  return out;
}
