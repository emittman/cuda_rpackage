#include "../header/gibbs.h"

/*struct modify_gamma_par_w_int: thrust::binary_function<double, int, double>{
  double operator()(double p, int x){
    return p + x/2.0;
  }
};

struct modify_gamma_par_w_flt: thrust::binary_function<double, double, double>{
  double operator()(double p, double x){
    return p + x/2;
  }
};*/

struct modify_gamma_par {
  template<typename T>
  void operator()(T tup){
    thrust::get<0>(tup) = thrust::get<0>(tup) + 1.0;/// 2.0 * thrust::get<1>(tup);
  }
};

void draw_tau2(curandState *states, chain_t &chain, priors_t &priors, data_t &data, summary2 &smry){
  fvec_d sse(smry.num_occupied);
  int K = chain.K;
  smry.sumSqErr(sse, chain.beta, data.xtx);
  std::cout << "sse:\n";
  printVec(sse, K, 1);
  fvec_d a_tmp(K, priors.a);
  fvec_d b_tmp(K, priors.b);
  std::cout << "a filled:\n";
  printVec(a_tmp, K, 1);
  // modify gamma parameters for occupied clusters
  //typedef thrust::permutation_iterator<intIter, intIter> IntPermIter;
  //typedef thrust::permutation_iterator<realIter, intIter> FltPermIter;
  typedef thrust::tuple<realIter, intIter> tuple1;
  typedef thrust::zip_iterator<tuple1> zip1;
  tuple1 tup1 = thrust::tuple<realIter, intIter>(a_tmp.begin(), smry.Mk.begin());
  zip1 zp1 = thrust::zip_iterator<tuple1>(tup1);
  
  modify_gamma_par f;
  thrust::for_each(zp1, zp1 + K, f);
  
  //IntPermIter Mk_iter =  thrust::permutation_iterator<intIter, intIter>(smry.Mk.begin(), smry.occupied.begin());
  //std::cout << "I thought Mk was length " << smry.Mk.size() << "\n";
  //std::cout << "and that a was length " << a.size() << " \n";
  //printVec(smry.Mk, K, 1);
  
  //intIter Mk_iter = smry.Mk.begin();
  //realIter a_begin = a.begin();
  //modify_gamma_par_w_flt f;
  //thrust::transform(a_begin, a_begin + 1, Mk_iter, a_begin, f);
  
  std::cout << "a transformed:\n";

  printVec(a_tmp, K, 1);
  //intIter occ_begin = smry.occupied.begin();
  //FltPermIter b_occ = thrust::permutation_iterator<realIter, intIter>(b.begin(), occ_begin);
  //thrust::transform(b_occ, b_occ + smry.num_occupied, sse.begin(), b_occ, modify_gamma_par_w_flt());
  std::cout << "b transformed:\n";
  printVec(b_tmp, K, 1);
  // raw pointers
  //double *tau2_ptr = thrust::raw_pointer_cast(chain.tau2.data());
  //double *a_ptr = thrust::raw_pointer_cast(a.data());
  //double *b_ptr = thrust::raw_pointer_cast(b.data());
  
  //generate
  //getGamma<<<K, 1>>>(states, a_ptr, b_ptr, tau2_ptr);
}
