#include "../header/summary2.h"


struct is_zero: thrust::unary_function<int, bool>{
  __host__ __device__ bool operator()(int i){
    return i==0 ? 1 : 0;
  }
};

// zeta passed by value, data passed by reference
summary2::summary2(int _K, ivec_d zeta, data_t &data): G(data.G), K(_K), V(data.V), occupied(_K), Mk(_K, 0){
  
  // local allocations
  ivec_d perm(data.G); // will store permutation
  thrust::sequence(perm.begin(), perm.end(), 0, 1);
  
  // sort, identify occupied
  thrust::sort_by_key(zeta.begin(), zeta.end(), perm.begin());
  ivec_d::iterator endptr;
  endptr = thrust::unique_copy(zeta.begin(), zeta.end(), occupied.begin());
  occupied.erase(endptr, occupied.end());
  num_occupied = occupied.size();
  
  // calculate Mk
  thrust::permutation_iterator<intIter, intIter> mapMk = thrust::permutation_iterator<intIter,intIter>(Mk.begin(), occupied.begin());
  thrust::reduce_by_key(zeta.begin(), zeta.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), mapMk);
  
  // unoccupied
  unoccupied.reserve(K - num_occupied);
  endptr = thrust::copy_if(thrust::make_counting_iterator<int>(0),
                           thrust::make_counting_iterator<int>(K), Mk.begin(), unoccupied.begin(), is_zero());
  unoccupied.erase(endptr, unoccupied.end());
  
  //size vectors
  yty_sums.reserve(num_occupied);
  xty_sums.reserve(num_occupied*V);
  ytx_sums.reserve(num_occupied*V);
  
  /*yty_sums
   *
   */
  // "arrange" data
  thrust::permutation_iterator<realIter, intIter> sort_yty = thrust::permutation_iterator<realIter, intIter>(data.yty.begin(), perm.begin());
  thrust::reduce_by_key(zeta.begin(), zeta.end(), sort_yty, thrust::make_discard_iterator(), yty_sums.begin());
  
  /* ytx_sums
   * 
   */
  // broadcast key
  RSIntIter zeta_rep = getRSIntIter(zeta.begin(), zeta.end(), K);
  // "arrange" data
  RSIntIter in_index = getRSIntIter(perm.begin(),perm.end(), G);
  thrust::permutation_iterator<realIter, RSIntIter> sort_ytx = thrust::permutation_iterator<realIter, RSIntIter>(data.ytx.begin(), in_index);
  // reduce
  thrust::reduce_by_key(zeta_rep, zeta_rep + G*V, sort_ytx, thrust::make_discard_iterator(), ytx_sums.begin());
  
  /* xty_sums
  * 
    */
    transpose<realIter>(ytx_sums.begin(), ytx_sums.end(), num_occupied, V, xty_sums.begin());
  
}

void summary2::draw_MVNormal(curandState *states, fvec_d &beta_hat, fvec_d &chol_prec, fvec_d &beta, priors_t &priors){
  
  //replace current beta with standard normal draws
  getNormal<<<K, V>>>(states, thrust::raw_pointer_cast(beta.data()));
  
  std::cout << "N(0,1) draws:\n";
  printVec(beta, V, K);
  
  //scale occupied betas by t(chol_prec)^-1
  scale_chol_inv(chol_prec, beta, occupied, num_occupied, V);

  std::cout << "scaled draws:\n";
  printVec(beta, V, K);
  
  //typedef: iterate along select columns of matrix of doubles
  typedef thrust::permutation_iterator<realIter, SCIntIter> gSCIter;
  
  //need access to occupied betas
  SCIntIter occ_idx = getSCIntIter(occupied.begin(), occupied.end(), V);
  gSCIter betaOcc = thrust::permutation_iterator<realIter, SCIntIter>(beta.begin(), occ_idx);
  
  //shift occupied betas by beta_hat
  thrust::transform(beta_hat.begin(), beta_hat.end(), betaOcc, betaOcc, thrust::plus<double>());
  
  std::cout << "occupied draws:\n";
  printVec(beta, V, K);
  
  //now, access to unoccupied betas
  SCIntIter unocc_idx = getSCIntIter(unoccupied.begin(), unoccupied.end(), V);
  gSCIter betaUnocc = thrust::permutation_iterator<realIter, SCIntIter>(beta.begin(), unocc_idx);
  
  //repeat prior var. and mean
  gRepTimes<realIter>::iterator prior_vars = getGRepTimesIter(priors.lambda2.begin(), priors.lambda2.end(), V);
  gRepTimes<realIter>::iterator prior_mean = getGRepTimesIter(priors.mu0.begin(), priors.mu0.end(), V);
  
  typedef thrust::tuple<gSCIter, gRepTimes<realIter>::iterator> linTransSomeBeta_tup;
  typedef thrust::zip_iterator<linTransSomeBeta_tup> linTransSomeBeta_zip;
  
  int num_unoccupied = K - num_occupied;
  
  //scale by prior sd
  linTransSomeBeta_tup scale_tup2 = thrust::tuple<gSCIter, gRepTimes<realIter>::iterator>(betaUnocc, prior_vars);
  linTransSomeBeta_zip scale_zip2 = thrust::zip_iterator<linTransSomeBeta_tup>(scale_tup2);
  mult_scalar_by_sqrt f2;
  thrust::for_each(scale_zip2, scale_zip2 + num_unoccupied*V, f2);
  
  std::cout << "unoccupied are scaled now:\n";
  printVec(beta, V, K);
  //shift by prior mean
  thrust::transform(prior_mean, prior_mean + num_unoccupied*V, betaUnocc, betaUnocc, thrust::plus<double>());
  std::cout << "and shifted (final draws):\n";
  printVec(beta, V, K);
}