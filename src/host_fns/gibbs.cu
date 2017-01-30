#include "../header/gibbs.h"
struct modify_gamma_par_w_int{
  double operator()(double p, int x){
    return p + x/2;
  }
};

struct modify_gamma_par_w_flt{
  double operator()(double p, double x){
    return p + x/2;
  }
};

void draw_tau2(curandState *states, chain_t &chain, priors_t &priors, data_t &data, summary2 &smry){
  fvec_d sse(smry.num_occupied);
  smry.sumSqErr(sse, chain.beta, data.xtx);
  fvec_d a(priors.K, priors.a);
  fvec_d b(priors.K, priors.b);
  std::cout << "a filled:\n";
  printVec(a, priors.K, 1);
  // modify gamma parameters for occupied clusters
  //typedef thrust::permutation_iterator<intIter, intIter> IntPermIter;
  typedef thrust::permutation_iterator<realIter, intIter> FltPermIter;
  //IntPermIter Mk_iter =  thrust::permutation_iterator<intIter, intIter>(smry.Mk.begin(), smry.occupied.begin());
  std::cout << "I thought Mk was length " << smry.Mk.size() << "\n";
  std::cout << "and that a was length " << a.size() << " \n";
  printVec(smry.Mk, smry.K, 1);
  //thrust::transform(a.begin(), a.end(), smry.Mk.begin(), a.begin(), modify_gamma_par_w_int());
  std::cout << "a transformed:\n";

  printVec(a, priors.K, 1);
  FltPermIter b_occ = thrust::permutation_iterator<realIter, intIter>(b.begin(), smry.occupied.begin());
  thrust::transform(b_occ, b_occ + smry.num_occupied, sse.begin(), b_occ, modify_gamma_par_w_flt());
  std::cout << "b transformed:\n";
  printVec(b, priors.K, 1);
  // raw pointers
  double *tau2_ptr = thrust::raw_pointer_cast(chain.tau2.data());
  double *a_ptr = thrust::raw_pointer_cast(a.data());
  double *b_ptr = thrust::raw_pointer_cast(b.data());
  /*
  // generate
  getGamma<<<chain.K, 1>>>(states, a_ptr, b_ptr, tau2_ptr);*/
}
