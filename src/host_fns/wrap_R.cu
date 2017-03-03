#include "../header/wrap_R.h"

data_t Rdata_wrap(SEXP Rdata){
  double *yty = REAL(VECTOR_ELT(Rdata, 0)),
    *xty = REAL(VECTOR_ELT(Rdata, 1)),
    *xtx = REAL(VECTOR_ELT(Rdata, 2));
  int G = INTEGER(VECTOR_ELT(Rdata, 3))[0],
      V = INTEGER(VECTOR_ELT(Rdata, 4))[0],
      N = INTEGER(VECTOR_ELT(Rdata, 5))[0];
  data_t data(yty, xty, xtx, G, V, N);
  return data;
}

priors_t Rpriors_wrap(SEXP Rpriors){
  int K = INTEGER(VECTOR_ELT(Rpriors, 0))[0],
      V = INTEGER(VECTOR_ELT(Rpriors, 1))[0];
  double *mu0 = REAL(VECTOR_ELT(Rpriors, 2)),
         *lambda = REAL(VECTOR_ELT(Rpriors, 3)),
         alpha = REAL(VECTOR_ELT(Rpriors, 4))[0],
         a = REAL(VECTOR_ELT(Rpriors, 5))[0],
         b = REAL(VECTOR_ELT(Rpriors, 6))[0];
  priors_t priors(K, V, mu0, lambda, alpha, a, b);
  return priors;
}

chain_t Rchain_wrap(SEXP Rchain){
  int G = INTEGER(VECTOR_ELT(Rchain, 0))[0],
      V = INTEGER(VECTOR_ELT(Rchain, 1))[0],
      K = INTEGER(VECTOR_ELT(Rchain, 2))[0],
      P = INTEGER(VECTOR_ELT(Rchain, 3))[0];
  double *beta = REAL(VECTOR_ELT(Rchain, 4)),
         *pi   = REAL(VECTOR_ELT(Rchain, 5)),
         *tau2 = REAL(VECTOR_ELT(Rchain, 6));
  int    *zeta = INTEGER(VECTOR_ELT(Rchain, 7));
  double *C = REAL(VECTOR_ELT(Rchain, 8));
  double *probs = REAL(VECTOR_ELT(Rchain, 9)),
         *means = REAL(VECTOR_ELT(Rchain, 10)),
         *meansquares = REAL(VECTOR_ELT(Rchain, 11));
  chain_t chain(G, V, K, P, beta, pi, tau2, zeta, C, probs, means, meansquares);
  return chain;
}

SEXP Csamples_wrap(samples_t &samples){
  SEXP samples_out      = PROTECT(allocVector(VECSXP, 5));
  SEXP out_beta         = PROTECT(allocVector(REALSXP, samples.save_beta.size()));
  SEXP out_tau2         = PROTECT(allocVector(REALSXP, samples.save_tau2.size()));
  SEXP out_P            = PROTECT(allocVector(REALSXP, samples.save_P.size()));
  SEXP out_max_id       = PROTECT(allocVector(INTSXP, samples.save_max_id.size()));
  SEXP out_num_occupied = PROTECT(allocVector(INTSXP, samples.save_num_occupied.size()));
  thrust::copy(samples.save_beta.begin(), samples.save_beta.end(), REAL(out_beta));
  thrust::copy(samples.save_tau2.begin(), samples.save_tau2.end(), REAL(out_tau2));
  thrust::copy(samples.save_P.begin(), samples.save_P.end(), REAL(out_P));
  thrust::copy(samples.save_max_id.begin(), samples.save_max_id.end(), INTEGER(out_max_id));
  thrust::copy(samples.save_num_occupied.begin(), samples.save_num_occupied.end(), INTEGER(out_num_occupied));
  SET_VECTOR_ELT(samples_out, 0, out_beta);
  SET_VECTOR_ELT(samples_out, 1, out_tau2);
  SET_VECTOR_ELT(samples_out, 2, out_P);
  SET_VECTOR_ELT(samples_out, 3, out_max_id);
  SET_VECTOR_ELT(samples_out, 4, out_num_occupied);
  return samples_out;
}

SEXP Cchain_wrap(chain_t &chain){
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
  return chain_out;
}
