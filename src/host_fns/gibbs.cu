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
  __host__ __device__ void operator()(T tup){
    thrust::get<0>(tup) = thrust::get<0>(tup) + 1.0/ 2.0 * thrust::get<1>(tup);
  }
};

void draw_tau2(curandState *states, chain_t &chain, priors_t &priors, data_t &data, summary2 &smry){
  fvec_d sse(smry.num_occupied);
  int K = chain.K;
  smry.sumSqErr(sse, chain.beta, data.xtx);
  std::cout << "sse:\n";
  printVec(sse, K, 1);
  fvec_d a_d(K, priors.a);
  fvec_d b_d(K, priors.b);
  std::cout << "a_d filled:\n";
  printVec(a_d, K, 1);
  // modify gamma parameters for occupied clusters
  typedef thrust::tuple<realIter, intIter> tuple1;
  typedef thrust::zip_iterator<tuple1> zip1;
  tuple1 tup1 = thrust::tuple<realIter, intIter>(a_d.begin(), smry.Mk.begin());
  zip1 zp1 = thrust::zip_iterator<tuple1>(tup1);
  
  modify_gamma_par f;
  thrust::for_each(zp1, zp1 + K, f);
  
  std::cout << "a transformed:\n";
  printVec(a_d, K, 1);
  
  typedef thrust::permutation_iterator<realIter, intIter> FltPermIter;
  FltPermIter b_occ = thrust::permutation_iterator<realIter, intIter>(b_d.begin(), smry.occupied.begin());
  typedef thrust::tuple<FltPermIter, realIter> tuple2;
  typedef thrust::zip_iterator<tuple2> zip2;
  tuple2 tup2 = thrust::tuple<FltPermIter, realIter>(b_occ, sse.begin());
  zip2 zp2 = thrust::zip_iterator<tuple2>(tup2);
  thrust::for_each(zp2, zp2 + K, modify_gamma_par());
  
  std::cout << "b transformed:\n";
  printVec(b_d, K, 1);
  // raw pointers
  double *tau2_ptr = thrust::raw_pointer_cast(chain.tau2.data());
  double *a_ptr = thrust::raw_pointer_cast(a_d.data());
  double *b_ptr = thrust::raw_pointer_cast(b_d.data());
  
  //generate
  getGamma<<<K, 1>>>(states, a_ptr, b_ptr, tau2_ptr);
}

void draw_pi(curandState *states, chain_t &chain, priors_t &priors, summary2 &summary){
  ivec_d Tk(priors.K);
  typedef thrust::reverse_iterator<intIter> revIntIter;
  revIntIter Tk_rev = thrust::reverse_iterator<int>(Tk.end());
  revIntIter Mk_rev = thrust::reverse_iterator<int>(summary.Mk.end());
  thrust::exclusive_scan(Mk_rev, Mk_rev + priors.K, Tk_rev);
  printVec(Tk, priors.K, 1);
}
