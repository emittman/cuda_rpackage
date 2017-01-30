#include "../header/gibbs.h"

struct modify_gamma_par{
  template<typename T>
  double operator()(double p, T x){
    return p + x/2;
  }
};

void draw_tau2(curandState *states, chain_t &chain, priors_t &prior, data_t &data, summary2 &smry){
  fvec_d sse(smry.num_occupied);
  smry.sumSqErr(sse, chain.beta, data.xtx);
  fvec_d a(prior.K, prior.a);
  fvec_d b(prior.K, prior.b);
  
  // modify gamma parameters for occupied clusters
  typedef thrust::permutation_iterator<intIter, intIter> IntPermIter;
  typedef thrust::permutation_iterator<realIter, intIter> FltPermIter;
  IntPermIter Mk_iter =  thrust::permutation_iterator<intIter, intIter>(smry.Mk.begin(), smry.occupied.begin());
  thrust::transform(a.begin(), a.end(), Mk_iter, a.begin(), modify_gamma_par());
  FltPermIter sse_iter = thrust::permutation_iterator<realIter, intIter>(sse.begin(), smry.occupied.begin());
  thrust::transform(b.begin(), b.end(), sse_iter, b.begin(), modify_gamma_par());
  
  // raw pointers
  double *tau2_ptr = thrust::raw_pointer_cast(chain.tau2.data());
  double *a_ptr = thrust::raw_pointer_cast(a.data());
  double *b_ptr = thrust::raw_pointer_cast(b.data());
  
  // generate
  getGamma<<<chain.K, 1>>>(states, a_ptr, b_ptr, tau2_ptr);
}
