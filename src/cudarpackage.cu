#include "header/cuda_usage.h"
#include "header/cholesky.h"
#include "header/quad_form.h"
#include "header/summary2.h"
#include "header/chain.h"
#include "header/iterator.h"
#include "header/construct_prec.h"
#include "header/distribution.h"
#include "header/cluster_probability.h"
#include "header/printing.h"
#include "header/gibbs.h"
#include "header/wrap_R.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
// This prevents the replacement of "beta" by Rmath.h
#ifdef beta
#undef beta
#endif
#include <boost/progress.hpp>

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

extern "C" SEXP Rconstruct_prec(SEXP Rdata, SEXP Rpriors, SEXP Rchain){
  data_t data = Rdata_wrap(Rdata);
  priors_t priors = Rpriors_wrap(Rpriors);
  chain_t chain = Rchain_wrap(Rchain);
  
  int psize = priors.K * data.V * data.V;
  summary2 summary(priors.K, chain.zeta, data);
  fvec_d prec(psize);
  construct_prec(prec, data, priors, chain, summary.Mk, 0);

  SEXP out_prec = PROTECT(allocVector(REALSXP, psize));
  for(int i=0; i<psize; ++i)
    REAL(out_prec)[i] = prec[i];
  UNPROTECT(1);
  return out_prec;
}

extern "C" SEXP Rgamma_rng(SEXP Rseed, SEXP a, SEXP b, bool logscale=false){

  int n = length(a), seed = INTEGER(Rseed)[0];
  
  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, n * sizeof(curandState)));
  
  //temporary memory
  
  fvec_d out_d(n);
  fvec_h out_h(n);
 
  fvec_d a_d(REAL(a), REAL(a)+n);
  fvec_d b_d(REAL(b), REAL(b)+n);
  
  double *out_d_ptr = thrust::raw_pointer_cast(out_d.data());
  double *a_d_ptr = thrust::raw_pointer_cast(a_d.data());
  double *b_d_ptr = thrust::raw_pointer_cast(b_d.data());
    
  //set up RNGs
  int n_blocks = n/512 + 1;
  setup_kernel<<<n_blocks,512>>>(seed, n, devStates);
  
  //sample from Gamma(a, b)
  getGamma<<<n_blocks,512>>>(devStates, n, a_d_ptr, b_d_ptr, out_d_ptr, logscale);
  
  //copy to host
  thrust::copy(out_d.begin(), out_d.end(), out_h.begin());
  
  //transfer memory
  SEXP out = PROTECT(allocVector(REALSXP, n));
  for(int i=0; i<n; ++i)
    REAL(out)[i] = out_h[i];
  
  //clean up
  CUDA_CALL(cudaFree(devStates));
  UNPROTECT(1);
  
  return out;
}

extern "C" SEXP Rbeta_rng(SEXP Rseed, SEXP a, SEXP b){

  int n = length(a), seed = INTEGER(Rseed)[0];

  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, n * sizeof(curandState)));
  
  //temporary memory
  
  fvec_d out_d(n);
  fvec_h out_h(n);
 
  fvec_d a_d(REAL(a), REAL(a)+n);
  fvec_d b_d(REAL(b), REAL(b)+n);
  
  double *out_d_ptr = thrust::raw_pointer_cast(out_d.data());
  double *a_d_ptr = thrust::raw_pointer_cast(a_d.data());
  double *b_d_ptr = thrust::raw_pointer_cast(b_d.data());
    
  //set up RNGs
  int blocksize = 512;
  int n_blocks = n/blocksize + 1;
  setup_kernel<<<n_blocks,blocksize>>>(seed, n, devStates);
  
  //sample from Beta(a, b)
  getBeta<<<n_blocks,blocksize>>>(devStates, n, a_d_ptr, b_d_ptr, out_d_ptr, false);
  
  //copy to host
  thrust::copy(out_d.begin(), out_d.end(), out_h.begin());
  
  //transfer memory
  SEXP out = PROTECT(allocVector(REALSXP, n));
  for(int i=0; i<n; ++i)
    REAL(out)[i] = out_h[i];
  
  //clean up
  CUDA_CALL(cudaFree(devStates));
  UNPROTECT(1);
  
  return out;
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
  fvec_h xtx(v*v, 1.0);
  double *ytyp = REAL(ytyR);
  double *ytxp = REAL(ytxR);
  double *xtyp = REAL(xtyR);
  double *xtxp = &(xtx[0]);
  data_t data(ytyp, xtyp, xtxp, g, v, 1);
  
  ivec_d ZETA(zp, zp+g);
  summary2 smry(k, ZETA, data);
  
  /*smry.print_Mk();
  smry.print_yty();
  smry.print_xty();*/
  
  SEXP out = PROTECT(allocVector(VECSXP, 4));
  SEXP OCCo = PROTECT(allocVector(INTSXP, 1));
  SEXP ytyo = PROTECT(allocVector(REALSXP, smry.num_occupied));
  SEXP ytxo = PROTECT(allocVector(REALSXP, smry.num_occupied*v));
  SEXP xtyo = PROTECT(allocVector(REALSXP, smry.num_occupied*v));
  INTEGER(OCCo)[0] = smry.num_occupied;
  
  for(int i=0; i<smry.num_occupied; ++i){
    REAL(ytyo)[i] = smry.yty_sums[i];
  }
  
  for(int i=0; i<smry.num_occupied*v; ++i){
    REAL(ytxo)[i] = smry.ytx_sums[i];
    REAL(xtyo)[i] = smry.xty_sums[i];
  }
  SET_VECTOR_ELT(out, 0, OCCo);
  SET_VECTOR_ELT(out, 1, ytyo);
  SET_VECTOR_ELT(out, 2, ytxo);
  SET_VECTOR_ELT(out, 3, xtyo);
  UNPROTECT(5);
  return out;
}

extern "C" SEXP Rdevice_mmultiply(SEXP AR, SEXP BR, SEXP a1R, SEXP a2R, SEXP b1R, SEXP b2R){
  int a1 = INTEGER(a1R)[0], a2 = INTEGER(a2R)[0], b1 = INTEGER(b1R)[0], b2 = INTEGER(b2R)[0];
  fvec_d A(REAL(AR), REAL(AR) + a1*a2), B(REAL(BR), REAL(BR) + b1*b2);
  fvec_d big_grid(a2*b2);
  big_matrix_multiply(A, B, big_grid, a1, a2, b1, b2);
  fvec_h big_grid_h(a2*b2);
  thrust::copy(big_grid.begin(), big_grid.end(), big_grid_h.begin());
  SEXP out = PROTECT(allocVector(REALSXP, a2*b2));
  for(int i=0; i<a2*b2; ++i) REAL(out)[i] = big_grid_h[i];
  UNPROTECT(1);
  return out;
}

extern "C" SEXP Rrun_mcmc(SEXP Rdata, SEXP Rpriors, SEXP RmethodPi, SEXP RmethodAlpha, SEXP Rchain, SEXP Rn_iter, SEXP Rn_save_P, SEXP Ridx_save, SEXP Rthin, SEXP Rseed, SEXP Rverbose){
  int verbose = INTEGER(Rverbose)[0];
  std::cout << "verbosity level = " << verbose << std::endl;
  data_t data      = Rdata_wrap(Rdata, verbose-1);
  priors_t priors  = Rpriors_wrap(Rpriors, verbose-1);
  chain_t chain    = Rchain_wrap(Rchain, verbose-1);
  int methodPi     = INTEGER(RmethodPi)[0],
      methodAlpha  = INTEGER(RmethodAlpha)[0],
      n_iter       = INTEGER(Rn_iter)[0],
      thin         = INTEGER(Rthin)[0],
      n_save_P     = INTEGER(Rn_save_P)[0];
  int G_save       = length(Ridx_save), seed = INTEGER(Rseed)[0];
  int n_save_g     = n_iter/thin + (n_iter % thin == 0 ? 0 : 1);
  bool alpha_fixed = methodAlpha == 0;
  
  /* Set thin_P to ensure at least n_save_P draws are saved*/
  int thin_P = n_iter - n_save_P; //in case n_save_P = 1, last iteration is saved
  
  if(n_save_P > 1){
    //if n_save_P is 2 or greater, thin_P is sup(x : x * n_save_P < n_iter)
    thin_P = n_iter/(n_save_P - 1) + (n_iter % (n_save_P - 1) == 0 ? -1 : 0);
  }
  
  samples_t samples(n_save_g, n_save_P, G_save, priors.K, data.V, INTEGER(Ridx_save), alpha_fixed);
  
  std::cout << "Model for pi: ";
  if(methodPi==0){
   std::cout << "Truncated stick-breaking process, ";
  } else if(methodPi==1){
   std::cout << "Symmetric Dirichlet distribution, ";
  }
  if(alpha_fixed){
    std::cout <<"alpha fixed" << std::endl;
  } else{
    std::cout <<"varying alpha" << std::endl;
  }
  
  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, data.G * data.V * sizeof(curandState)));
  setup_kernel<<<chain.G, chain.V>>>(seed, chain.G*chain.V, devStates);
  
  
  //progress bar
  boost::progress_display show_progress(n_iter);
  
  for(int i=0; i<n_iter; i++){
    //Gibbs steps
    summary2 summary(chain.K, chain.zeta, data);
    if(verbose > 1){
    std::cout << "Mk:\n";
    printVec(summary.Mk, priors.K, 1);
    std::cout << "occupied:\n";
    printVec(summary.occupied, summary.num_occupied, 1);
    std::cout << "unoccupied:\n";
    printVec(summary.unoccupied, priors.K - summary.num_occupied, 1);
    }
    
    draw_beta(devStates, data, chain, priors, summary, verbose-1);
    if(verbose > 1) {
      std::cout << "beta:\n";
      printVec(chain.beta, data.V, priors.K);
    }
    
    draw_tau2(devStates, chain, priors, data, summary, verbose-1);
    if(verbose > 1){
      std::cout << "tau2:\n";
      printVec(chain.tau2, priors.K, 1);
    }
    if(methodPi == 0){
      draw_pi(devStates, chain, priors, summary, verbose-1);
    } else if(methodPi == 1) {
      draw_pi_SD(devStates, chain, priors, summary, verbose-1);
    }
    if(verbose > 1) {
      std::cout << "pi:\n";
      printVec(chain.pi, priors.K, 1);
    }
    
    if(methodAlpha == 1){
      draw_alpha(chain, priors, verbose-1);
    }
    if(methodAlpha == 2){
      draw_alpha_SD(chain, priors, verbose-1);
    }
    if(!alpha_fixed & verbose > 0) {
      std::cout << "alpha = " << priors.alpha << std::endl;
    }
    
    draw_zeta(devStates, data, chain, priors, verbose-1);
    if(verbose > 1){
      std::cout << "zeta:\n";
      printVec(chain.zeta, data.G, 1);
    }
    
    if(i % thin == 0){
      if(!alpha_fixed){
        samples.save_alpha[samples.step_g] = priors.alpha;
      }
      samples.write_g_samples(chain, summary);
    }
    
    if(i % thin_P == 0 & samples.step_P < n_save_P){
      samples.write_P_samples(chain);
    }
    
    chain.update_means(i+1);
    chain.update_probabilities(i+1);
    ++show_progress;
  }
  
  CUDA_CALL(cudaFree(devStates));
  SEXP samples_out = Csamples_wrap(samples, verbose-1);          //PROTECT(7)
  SEXP chain_out   = Cchain_wrap(chain, verbose-1);              //PROTECT(4)
  SEXP out         = PROTECT(allocVector(VECSXP, 2)); //PROTECT(1)
  SET_VECTOR_ELT(out, 0, samples_out);
  SET_VECTOR_ELT(out, 1, chain_out);
  UNPROTECT(12);                                      //7 + 4 + 1
  
  return out;
}
