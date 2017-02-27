#include "../header/construct_prec.h"

__host__ __device__ void diagAdd::operator()(diag_tup_el Tup){
    thrust::get<0>(Tup) = thrust::get<0>(Tup) + thrust::get<1>(Tup)/thrust::get<2>(Tup);
  }

void construct_prec(realIter prec_begin, realIter prec_end, realIter lam_begin, realIter lam_end,
                    realIter tau_begin, realIter tau_end, intIter Mk_begin, intIter Mk_end,
                    realIter xtx_begin, realIter xtx_end, int K, int V){

  if(thrust::distance(prec_begin, prec_end) < K*V*V) std::cout <<"DIMENSION MISMATCH!\n";

  //copy xtx over in a repeating loop (initialization)
  gRepTimes<realIter>::iterator xtx_rep = getGRepTimesIter(xtx_begin, xtx_end, V*V, 1); 
  thrust::copy(xtx_rep, xtx_rep + K*V*V, prec_begin);
  
  //multiply by Mk
  gRepEach<intIter>::iterator Mk_rep = getGRepEachIter(Mk_begin, Mk_end, V*V, 1);
  transform(prec_begin, prec_end, Mk_rep, prec_begin, thrust::multiplies<double>());

  //modify diagonal according to prior prec/error prec
  diagAdd f;
  gDiagonal<realIter>::iterator prec_diag = getGDiagIter(prec_begin, prec_end, V);
  gRepTimes<realIter>::iterator lam_iter  = getGRepTimesIter(lam_begin, lam_end, V, 1);
  gRepEach<realIter>::iterator   tau_iter  = getGRepEachIter(tau_begin, tau_end, V, 1);
  diag_zip zipped = thrust::zip_iterator<diag_tup>(thrust::make_tuple(prec_diag, lam_iter, tau_iter));
  thrust::for_each(zipped, zipped+K*V, f);
  
  /*
  */

}

void construct_prior_weighted_mean(fvec_d &prior_w_mean, priors_t &priors, chain_t &chain){
  realIter lambda2_begin = priors.lambda2.begin();
  realIter lambda2_end   = priors.lambda2.end();
  gRepTimes<realIter>::iterator rep_lambda2 = getGRepTimesIter(lambda2_begin, lambda2_end, chain.V);
  
  realIter mu0_begin = priors.mu0.begin();
  realIter mu0_end   = priors.mu0.end();
  gRepTimes<realIter>::iterator rep_mu0 = getGRepTimesIter(mu0_begin, mu0_end, chain.V);
  
  realIter tau2_begin = chain.tau2.begin();
  realIter tau2_end   = chain.tau2.end();
  gRepEach<realIter>::iterator each_tau2 = getGRepEachIter(tau2_begin, tau2_end, chain.V);
  
  realIter p_w_mean = prior_w_mean.begin();
  
  mean_zip zip = thrust::zip_iterator<mean_tup>(thrust::make_tuple(rep_lambda2, each_tau2, rep_mu0, p_w_mean));
  weighted_prior_mean f;
  thrust::for_each(zip, zip + priors.K * chain.V, f);
}
