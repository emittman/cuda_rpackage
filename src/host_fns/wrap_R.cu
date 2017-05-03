#include "../header/wrap_R.h"

data_t Rdata_wrap(SEXP Rdata, int verbose){
  if(verbose>0){
    std::cout << "Reading data... ";
  }
  double *yty = REAL(VECTOR_ELT(Rdata, 0)),
    *xty = REAL(VECTOR_ELT(Rdata, 1)),
    *xtx = REAL(VECTOR_ELT(Rdata, 2));
  int G = INTEGER(VECTOR_ELT(Rdata, 3))[0],
      V = INTEGER(VECTOR_ELT(Rdata, 4))[0],
      N = INTEGER(VECTOR_ELT(Rdata, 5))[0];
  bool voom = LOGICAL(VECTOR_ELT(Rdata, 6))[0];
  data_t data(yty, xty, xtx, G, V, N, voom);
  if(verbose>0){
    std::cout << "data transferred." << std::endl;
  }
  return data;
}

priors_t Rpriors_wrap(SEXP Rpriors, int verbose){
  if(verbose>0){
    std::cout << "Reading priors... ";
  }
  int K = INTEGER(VECTOR_ELT(Rpriors, 0))[0],
      V = INTEGER(VECTOR_ELT(Rpriors, 1))[0];
  double *mu0 = REAL(VECTOR_ELT(Rpriors, 2)),
         *lambda = REAL(VECTOR_ELT(Rpriors, 3)),
         a = REAL(VECTOR_ELT(Rpriors, 4))[0],
         b = REAL(VECTOR_ELT(Rpriors, 5))[0],
         A = REAL(VECTOR_ELT(Rpriors, 6))[0],
         B = REAL(VECTOR_ELT(Rpriors, 7))[0];
  priors_t priors(K, V, mu0, lambda, a, b, A, B);
  if(verbose>0){
    std::cout << "priors transferred." << std::endl;
  }
  return priors;
}

chain_t Rchain_wrap(SEXP Rchain, int verbose){
  if(verbose>0){
    std::cout << "Reading chain... ";
  }
  int    G = INTEGER(VECTOR_ELT(Rchain, 0))[0],
         V = INTEGER(VECTOR_ELT(Rchain, 1))[0],
         K = INTEGER(VECTOR_ELT(Rchain, 2))[0],
     n_hyp = INTEGER(VECTOR_ELT(Rchain, 3))[0],
  *C_rowid = INTEGER(VECTOR_ELT(Rchain, 4)),
         P = INTEGER(VECTOR_ELT(Rchain, 5))[0];
  double *beta = REAL(VECTOR_ELT(Rchain, 6)),
         *pi   = REAL(VECTOR_ELT(Rchain, 7)),
         *tau2 = REAL(VECTOR_ELT(Rchain, 8));
  int    *zeta = INTEGER(VECTOR_ELT(Rchain, 9));
  double alpha = REAL(VECTOR_ELT(Rchain, 10))[0];
  double           *C = REAL(VECTOR_ELT(Rchain, 11)),
               *probs = REAL(VECTOR_ELT(Rchain, 12)),
               *means = REAL(VECTOR_ELT(Rchain, 13)),
         *meansquares = REAL(VECTOR_ELT(Rchain, 14)),
          slice_width = REAL(VECTOR_ELT(Rchain, 15))[0],
           max_steps  = REAL(VECTOR_ELT(Rchain, 16))[0];
  chain_t chain(G, V, K, n_hyp, C_rowid, P, beta, pi, tau2, zeta, alpha, C, probs, means, meansquares, slice_width, max_steps);
  if(verbose>0){
    std::cout << "chain transferred." << std::endl;
  }
  return chain;
}


SEXP Csamples_wrap(samples_t &samples, int verbose){
  if(verbose>0){
    std::cout << "Wrapping samples... ";
  }
  int size = 5;
  if(!samples.alpha_fixed) ++size;
  SEXP samples_out      = PROTECT(allocVector(VECSXP, size));
  SEXP out_beta         = PROTECT(allocVector(REALSXP, samples.save_beta.size()));
  SEXP out_tau2         = PROTECT(allocVector(REALSXP, samples.save_tau2.size()));
  SEXP out_P            = PROTECT(allocVector(REALSXP, samples.save_P.size()));
  SEXP out_max_id       = PROTECT(allocVector(INTSXP, samples.save_max_id.size()));
  SEXP out_num_occupied = PROTECT(allocVector(INTSXP, samples.save_num_occupied.size()));
  SEXP out_alpha        = PROTECT(allocVector(REALSXP, samples.save_alpha.size()));

  thrust::copy(samples.save_beta.begin(), samples.save_beta.end(), REAL(out_beta));
  thrust::copy(samples.save_tau2.begin(), samples.save_tau2.end(), REAL(out_tau2));
  thrust::copy(samples.save_P.begin(), samples.save_P.end(), REAL(out_P));
  thrust::copy(samples.save_max_id.begin(), samples.save_max_id.end(), INTEGER(out_max_id));
  thrust::copy(samples.save_num_occupied.begin(), samples.save_num_occupied.end(), INTEGER(out_num_occupied));
  if(!samples.alpha_fixed){
    thrust::copy(samples.save_alpha.begin(), samples.save_alpha.end(), REAL(out_alpha));
  }
  SET_VECTOR_ELT(samples_out, 0, out_beta);
  SET_VECTOR_ELT(samples_out, 1, out_tau2);
  SET_VECTOR_ELT(samples_out, 2, out_P);
  SET_VECTOR_ELT(samples_out, 3, out_max_id);
  SET_VECTOR_ELT(samples_out, 4, out_num_occupied);
  if(!samples.alpha_fixed){
    SET_VECTOR_ELT(samples_out, 5, out_alpha);
  }
  if(verbose>0){
    std::cout << "samples wrapped." << std::endl;
  }
  return samples_out;
}

SEXP Cchain_wrap(chain_t &chain, int verbose){
  if(verbose>0){
    std::cout << "Wrapping chain... ";
  }
  SEXP chain_out       = PROTECT(allocVector(VECSXP, 3));
  SEXP out_probs       = PROTECT(allocVector(REALSXP, chain.probs.size()));
  SEXP out_means       = PROTECT(allocVector(REALSXP, chain.means.size()));
  SEXP out_meansquares = PROTECT(allocVector(REALSXP, chain.meansquares.size()));
  thrust::copy(chain.probs.begin(), chain.probs.end(), REAL(out_probs));
  thrust::copy(chain.means.begin(), chain.means.end(), REAL(out_means));
  thrust::copy(chain.meansquares.begin(), chain.meansquares.end(), REAL(out_meansquares));
  SET_VECTOR_ELT(chain_out, 0, out_probs);
  SET_VECTOR_ELT(chain_out, 1, out_means);
  SET_VECTOR_ELT(chain_out, 2, out_meansquares);
  if(verbose>0){
    std::cout << "chain wrapped." << std::endl;
  }
  return chain_out;
}

SEXP Cstate_wrap(chain_t &chain, int verbose){
  if(verbose>0){
    std::cout << "Wrapping state... ";
  }
  SEXP state_out = PROTECT(allocVector(VECSXP, 4));
  SEXP out_beta  = PROTECT(allocVector(REALSXP, chain.beta.size()));
  SEXP out_tau2  = PROTECT(allocVector(REALSXP, chain.tau2.size()));
  SEXP out_pi    = PROTECT(allocVector(REALSXP, chain.pi.size()));
  SEXP out_alpha = PROTECT(allocVector(REALSXP, 1));
  thrust::copy(chain.beta.begin(), chain.beta.end(), REAL(out_beta));
  thrust::copy(chain.tau2.begin(), chain.tau2.end(), REAL(out_tau2));
  thrust::copy(chain.pi.begin(), chain.pi.end(), REAL(out_pi));
  REAL(out_alpha)[0] = chain.alpha;
  SET_VECTOR_ELT(state_out, 0, out_beta);
  SET_VECTOR_ELT(state_out, 1, out_tau2);
  SET_VECTOR_ELT(state_out, 2, out_pi);
  SET_VECTOR_ELT(state_out, 3, out_alpha);
  if(verbose>0){
    std::cout << "state wrapped." << std::endl;
  }
  return state_out;
}
