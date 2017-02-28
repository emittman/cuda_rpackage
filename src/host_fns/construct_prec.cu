#include "../header/construct_prec.h"

__host__ __device__ void diagAdd::operator()(diag_tup_el Tup){
    thrust::get<0>(Tup) = thrust::get<0>(Tup) + thrust::get<1>(Tup);
  }

void construct_prec(fvec_d &prec, data_t &data, priors_t &priors, chain_t &chain, ivec_d &Mk, int verbose = 0){
  int K = priors.K, V = data.V;
  if(prec.size() < K*V*V) std::cout <<"DIMENSION MISMATCH!\n";

  //iterators for reuse
  realIter prec_begin = prec.begin();
  realIter prec_end   = prec.end();

  //copy xtx over in a repeating loop (initialization)
  realIter xtx_begin = data.xtx.begin();
  realIter xtx_end = data.xtx.end();
  gRepTimes<realIter>::iterator xtx_rep = getGRepTimesIter(xtx_begin, xtx_end, V*V, 1); 
  thrust::copy(xtx_rep, xtx_rep + K*V*V, prec_begin);
  
  //multiply by Mk[k], tau2[k]
  intIter Mk_begin = Mk.begin();
  intIter Mk_end   = Mk.end();
  gRepEach<intIter>::iterator Mk_rep = getGRepEachIter(Mk_begin, Mk_end, V*V, 1);
  gRepEach<realIter>::iterator tau2_rep = getGRepEachIter(chain.tau2.begin(), chain.tau2.end(), V*V, 1);
  transform(prec_begin,prec_end, Mk_rep, prec_begin, thrust::multiplies<double>());
  if(verbose>0){
    std::cout << "xty_sums * Mk" << std::endl;
    printVec(prec, V, K);
  }
  transform(prec_begin, prec_end, tau2_rep, prec_begin, thrust::multiplies<double>());
  if(verbose>0){
    std::cout << "xty_sums * Mk * tau2" << std::endl;
    printVec(prec, V, K);
  }
  //modify diagonal; increment by prior prec
  diagAdd f;
  gDiagonal<realIter>::iterator prec_diag = getGDiagIter(prec_begin, prec_end, V);
  gRepTimes<realIter>::iterator lambda2_iter  = getGRepTimesIter(priors.lambda2.begin(), priors.lambda2.end(), V, 1);
  diag_zip zipped = thrust::zip_iterator<diag_tup>(thrust::make_tuple(prec_diag, lambda2_iter));
  thrust::for_each(zipped, zipped+K*V, f);

}

__host__ __device__ void weighted_sum_functor::operator()(wt_sum_el tup){
  thrust::get<0>(tup) = thrust::get<0>(tup) * thrust::get<1>(tup) + thrust::get<2>(tup) * thrust::get<3>(tup);
}

void construct_weighted_sum(fvec_d &weighted_sum, summary2 &smry, priors_t &priors, chain_t &chain, int verbose=0){
  //return xty_sums[k] * tau2[k] + mu_0 * lambda2
  realIter wt_sum_begin = weighted_sum.begin();
  realIter wt_sum_end   = weighted_sum.end();
  
  SCIntIter colIter = getSCIntIter(smry.occupied.begin(), smry.occupied.end(), chain.V);
  if(verbose>0){
    std::cout << "Testing colIter:\n";
    thrust::copy(colIter, colIter + smry.num_occupied * chain.V, std::ostream_iterator<int>(std::cout, " "));
  }
  typedef thrust::permutation_iterator<realIter, SCIntIter> gColIter;
  gColIter clustOcc = thrust::permutation_iterator<realIter, SCIntIter>(weighted_sum.begin(), colIter);
  thrust::copy(smry.xty_sums.begin(), smry.xty_sums.end(), clustOcc);
  if(verbose>0){
    std::cout << "xty_sums:\n";
    printVec(smry.xty_sums, chain.V, smry.num_occupied);
    std::cout << "xty_sums mapped to clusters:\n";
    printVec(weighted_sum, chain.V, priors.K);
  }
  realIter tau2_begin = chain.tau2.begin();
  realIter tau2_end   = chain.tau2.end();
  gRepEach<realIter>::iterator each_tau2 = getGRepEachIter(tau2_begin, tau2_end, chain.V);

  realIter lambda2_begin = priors.lambda2.begin();
  realIter lambda2_end   = priors.lambda2.end();
  gRepTimes<realIter>::iterator rep_lambda2 = getGRepTimesIter(lambda2_begin, lambda2_end, chain.V);
  
  realIter mu0_begin = priors.mu0.begin();
  realIter mu0_end   = priors.mu0.end();
  gRepTimes<realIter>::iterator rep_mu0 = getGRepTimesIter(mu0_begin, mu0_end, chain.V);
  
  wt_sum_zip zip = thrust::zip_iterator<wt_sum_tup>(thrust::make_tuple(wt_sum_begin, each_tau2, rep_mu0, rep_lambda2));
  weighted_sum_functor g;
  thrust::for_each(zip, zip + priors.K * chain.V, g);
}
