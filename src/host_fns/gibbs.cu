#include "../header/gibbs.h"

struct modify_gamma_par{
  template<typename T>
  double operator()(double p, T x){
    return p + x/2;
  }
}

void draw_tau2(curandState *states, chain_t &chain, prior_t &prior, data &data, summary2 &smry){
  fvec_d sse(smry.num_occupied);
  sumSqErr(sse, chain.beta, data.xtx);
  fvec_d a(prior.K, prior.a);
  fvec_d b(prior.K, prior.b);
  
  // modify gamma parameters for occupied clusters
  typedef thrust::permutation_iterator<intIter, intIter> IntPermIter;
  typedef thrust::permutation_iterator<realIter, intIter> FltPermIter;
  IntPermIter Mk_iter =  thrust::permutation_iterator<intIter, intIter>(Mk.begin(), occupied.begin());
  thrust::transform(a.begin(), a.end(), Mk_iter, a.begin(), modify_gamma_par());
  FltPermIter sse_iter = thrust::permutation_iterator<realIter, intIter>(sse.begin(), occupied.begin());
  thrust::transform(b.begin(), b.end(), sse_iter, b.begin(), modify_gamma_par());
  
  // raw pointers
  tau2_ptr = thrust::raw_pointer_cast(chain.tau2.data());
  a_ptr = thrust::raw_pointer_cast(a.data());
  b_ptr = thrust::raw_pointer_cast(b.data());
  
  // generate
  getGamma(states, a_ptr, b_ptr, tau2_ptr);
}
