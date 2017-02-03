#include "header/cuda_usage.h"
#include "header/chain.h"
#include "header/iterator.h"
#include "header/printing.h"
#include "header/cluster_probability.h"
#include "header/multinomial.h"
#include "header/distribution.h"
#include "header/beta_hat.h"
#include "header/wrap_R.h"
#include "header/summary2.h"
#include "header/cholesky.h"
#include "header/construct_prec.h"
#include "header/multi_dot_product.h"
#include "header/gibbs.h"
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <cuda.h>

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

extern "C" SEXP Rcluster_weights(SEXP A, SEXP B, SEXP C, SEXP D, SEXP E, SEXP G, SEXP N, SEXP K){
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
  
  //clean up
  CUDA_CALL(cudaFree(devStates));
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
  
  //clean up
  CUDA_CALL(cudaFree(devStates));
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


extern "C" SEXP Rtest_data_wrap(SEXP Rdata, SEXP Rpriors, SEXP Rchain){
  data_t data = Rdata_wrap(Rdata);
  priors_t priors = Rpriors_wrap(Rpriors);
  chain_t chain = Rchain_wrap(Rchain);
  std::cout << "y transpose x\n";
  printVec(data.ytx, data.G, data.V);
  std::cout << "prior location\n";
  printVec(priors.mu0, priors.V, 1);
  std::cout << "Contrasts on location parameters";
  printVec(chain.C, chain.P, chain.V);
  SEXP out = PROTECT(allocVector(INTSXP, 1));
  INTEGER(out)[0] = 0;
  UNPROTECT(1);
  return out;
}

extern"C" SEXP Rtest_MVNormal(SEXP Rseed, SEXP Rzeta, SEXP Rdata, SEXP Rpriors){
  int seed = INTEGER(Rseed)[0];
  data_t data = Rdata_wrap(Rdata);
  priors_t priors = Rpriors_wrap(Rpriors);
  ivec_h zeta_h(INTEGER(Rzeta), INTEGER(Rzeta) + data.G);
  ivec_d zeta_d(zeta_h.begin(),zeta_h.end());  
  
  summary2 smry(priors.K, zeta_d, data);
  
  smry.print_Mk();
  smry.print_yty();
  smry.print_xty();
  
  
  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, data.V*priors.K * sizeof(curandState)));
  setup_kernel<<<priors.K, data.V>>>(seed, devStates);
  
  
  //make precision matrices
  fvec_d prec(smry.num_occupied * smry.V * smry.V, 0.0);
  fvec_d tau2(priors.K, 1.0);
  construct_prec(prec.begin(), prec.end(), priors.lambda2.begin(), priors.lambda2.end(), tau2.begin(), tau2.end(),
                 smry.Mk.begin(), smry.Mk.end(), data.xtx.begin(), data.xtx.end(), smry.num_occupied, data.V);
  
  std::cout << "prec_matrices:\n";
  printVec(prec, smry.V*smry.V, smry.num_occupied);
  
  //cholesky decomposition
  realIter b=prec.begin(), e = prec.end();
  chol_multiple(b, e,  data.V, smry.num_occupied);
  
  std::cout << "chol_matrices:\n";
  printVec(prec, smry.V*smry.V, smry.num_occupied);
  
  std::cout << "smry.V= " << smry.V << "\n";
  std::cout << "smry.num_occupied= " << smry.num_occupied << "\n";
  
  //conditional means
  std::cout << "xty_sums:\n";
  thrust::device_ptr<double> xty_ptr = &smry.xty_sums[0];
  thrust::copy(xty_ptr, xty_ptr + smry.num_occupied * data.V, std::ostream_iterator<double>(std::cout, " "));  
  fvec_d bhat(smry.num_occupied * data.V);
  thrust::copy(xty_ptr, xty_ptr + smry.num_occupied * data.V, bhat.begin());
  cudaDeviceSynchronize();
  std::cout << "container for beta_hat (initialized):\n";
  printVec(bhat, data.V, smry.num_occupied);
  
  beta_hat(prec, bhat, smry.num_occupied, data.V);
  
  
  std::cout << "beta_hat:\n";
  printVec(bhat, data.V, smry.num_occupied);
  
  //draw beta
  int beta_size = data.V*priors.K;
  fvec_h beta_h(beta_size, 0.0);
  fvec_d beta(beta_h.begin(), beta_h.end());
  draw_MVNormal(devStates, bhat, prec, beta, priors, smry);
  
  //print value
  std::cout << "beta_draws:\n";
  printVec(beta, data.V, priors.K);
  
  thrust::copy(beta.begin(), beta.end(), beta_h.begin());
  
  SEXP out = PROTECT(allocVector(REALSXP, beta_size));

  for(int i=0; i < beta_size; ++i){
    REAL(out)[i] = beta_h[i];
  }
  
  //clean up
  CUDA_CALL(cudaFree(devStates));
  UNPROTECT(1);
  return out;
}

extern "C" SEXP Rmulti_dot_prod(SEXP Rx, SEXP Ry, SEXP Rdim, SEXP Rn){
  int dim = INTEGER(Rdim)[0], n = INTEGER(Rn)[0];
  fvec_h x_h(REAL(Rx), REAL(Rx) + dim*n);
  fvec_h y_h(REAL(Ry), REAL(Ry) + dim*n);
  fvec_d x_d(x_h.begin(), x_h.end());
  fvec_d y_d(y_h.begin(), y_h.end());
  fvec_d z_d(n);
  multi_dot_prod(x_d, y_d, z_d, dim, n);
  fvec_h z_h(n);
  thrust::copy(z_d.begin(), z_d.end(), z_h.begin());
  SEXP out = PROTECT(allocVector(REALSXP, n));
  for(int i = 0; i < n; ++i){
    REAL(out)[i] = z_h[i];
  }
  UNPROTECT(1);
  return out;
}

extern "C" SEXP RsumSqErr(SEXP Rdata, SEXP Rzeta, SEXP K, SEXP Rbeta){
  int k = INTEGER(K)[0];
  data_t data = Rdata_wrap(Rdata);
  std::cout << "\n G= " << data.G << "\n";
  std::cout << "\n xty:\n";
  printVec(data.xty, data.V, data.G);
  ivec_h zeta_h(INTEGER(Rzeta), INTEGER(Rzeta) + data.G);
  ivec_d zeta_d(zeta_h.begin(), zeta_h.end());
  summary2 smry(k, zeta_d, data);
  std::cout << "\nxty_sums:\n";
  printVec(smry.xty_sums, smry.V, smry.num_occupied);
  std::cout << "\nzeta_d:\n";
  printVec(zeta_d, data.G, 1);
  fvec_d beta(REAL(Rbeta), REAL(Rbeta) + smry.num_occupied*data.V);
  fvec_d sse_d(smry.num_occupied);
  smry.sumSqErr(sse_d, beta, data.xtx);
  fvec_h sse_h(smry.num_occupied);
  thrust::copy(sse_d.begin(), sse_d.end(), sse_h.begin());
  std::cout << "sse_d:\n";
  printVec(sse_d, smry.num_occupied, 1);
  thrust::device_ptr<double> sse_ptr = &sse_d[0];
  thrust::copy(sse_ptr, sse_ptr + smry.num_occupied, sse_h.begin());
  std::cout << "sse_h after:\n";
  printVec(sse_h, smry.num_occupied, 1);
  SEXP out = PROTECT(allocVector(REALSXP, smry.num_occupied));
  for(int i=0; i<smry.num_occupied; ++i){
    REAL(out)[i] = sse_h[i];
  }
  UNPROTECT(1);
  return out;
}

extern "C" SEXP Rtest_draw_tau2(SEXP Rseed, SEXP Rdata, SEXP Rchain, SEXP Rpriors){
  int seed = INTEGER(Rseed)[0];
  data_t data = Rdata_wrap(Rdata);
  chain_t chain = Rchain_wrap(Rchain);
  priors_t priors = Rpriors_wrap(Rpriors);
  summary2 smry(chain.K, chain.zeta, data);
 
  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, priors.K * sizeof(curandState)));
  setup_kernel<<<priors.K, 1>>>(seed, devStates);
 
  std::cout << "tau2 before:\n";
  printVec(chain.tau2, chain.K, 1);
 
  draw_tau2(devStates, chain, priors, data, smry);
  
  std::cout << "tau2 after:\n";
  printVec(chain.tau2, chain.K, 1);
  
  fvec_h tau2(chain.K);
  thrust::copy(chain.tau2.begin(), chain.tau2.end(), tau2.begin());
  
  SEXP out = PROTECT(allocVector(REALSXP, chain.K));
  for(int i=0; i<chain.K; ++i){
    REAL(out)[i] = tau2[i];
  }
  
  //clean up
  CUDA_CALL(cudaFree(devStates));
  UNPROTECT(1);
  return out;
}

extern "C" SEXP Rtest_draw_pi(SEXP Rseed, SEXP Rchain, SEXP Rpriors, SEXP Rdata){
  int seed = INTEGER(Rseed)[0];
  data_t data = Rdata_wrap(Rdata);
  chain_t chain = Rchain_wrap(Rchain);
  priors_t priors = Rpriors_wrap(Rpriors);
  summary2 smry(chain.K, chain.zeta, data);
  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, priors.K * sizeof(curandState)));
  setup_kernel<<<priors.K, 1>>>(seed, devStates);

  draw_pi(devStates, chain, priors, smry);
  
  SEXP out = PROTECT(allocVector(REALSXP, 1));
  REAL(out)[0] = 0;
  
  //clean up
  CUDA_CALL(cudaFree(devStates));
  UNPROTECT(1);
  return out;
}

