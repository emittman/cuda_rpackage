#include "../header/draw_MVNormal.h"

void summary2::draw_MVNormal(curandState *states, fvec_d &beta_hat, fvec_d &chol_prec, fvec_d &beta, priors_t &priors){
  typedef thrust::tuple<gSFRIter<realIter>::iterator, strideIter> scaleSomeBeta_tup;
  typedef thrust::zip_iterator<scaleSomeBeta_tup> scaleSomeBeta_zip;
  //replace current beta with standard normal draws
  getNormal<<<K, 1>>>(states, thrust::raw_pointer_cast(beta.data()));
  
  //need access to first elems of chols and occ. betas
  gSFRIter<realIter>::iterator betaOcc_first = getGSFRIter(beta.begin(), beta.end(), occupied, V);
  rowIter strides_idx = getRowIter(V*V, 0);
  strideIter L_first = thrust::permutation_iterator<realIter, rowIter>(chol_prec.begin(), strides_idx);
  scaleSomeBeta_tup scale_tup = thrust::make_tuple(betaOcc_first, L_first);
  scaleSomeBeta_zip scale_zip = thrust::zip_iterator<transSel_tup>(scale_tup);
  scale_vec f(V);
  
  //scale standard normals (occupied)
  thrust::for_each(scale_zip, scale_zip + num_occupied, f);
  
  typedef thrust::permutation_iterator<realIter, SCIntIter> gSCIter;
  
  //need access to all of occ. betas
  SCIntIter occ_idx = getSCIntIter(occupied.begin(), occupied.end(), V);
  gSCIter betaOcc = thrust::permutation_iterator<realIter SCIntIter>(beta.begin(), occ_idx);
  
  //shift scaled normals (occupied)
  thrust::transform(beta_hat.begin(), beta_hat.end(), betaOcc, betaOcc, thrust::plus<double>());
  
  //now, access to unoccupied betas
  SCIntIter unocc_idx = getSCIntIter(unoccupied.begin(), unoccupied.end(), V);
  gSCIter betaUnocc = thrust::permutation_iterator<realIter SCIntIter>(beta.begin(), unocc_idx);
  
  //repeat prior var. and mean
  gRepTimes<realIter>::iterator prior_vars = getGRepTimesIter(priors.lambda2.begin(), priors.lambda2.end(), V);
  gRepTimes<realIter>::iterator prior_mean = getGRepTimesIter(priors.mu0.begin(), priors.mu0.end(), V);
  
  typedef thrust::tuple<gSCIter, gRepTimes<realIter>::iterator> linTransSomeBeta_tup;
  typedef thrust::zip_iterator<linTransSomeBeta_tup> linTransSomeBeta_zip;
  
  int num_unoccupied = K - num_occupied;
  
  //scale by prior sd
  linTransSomeBeta_zip scale_tup2 = thrust::tuple<gSCIter, gRepTimes<realIter>::iterator>(betaUnocc, prior_vars);
  linTransSomeBeta_zip scale_zip2 = thrust::make_zip_iterator(scale_tup2);
  mult_scalar_by_sqrt f2;
  thrust::for_each(zip2, zip2 + num_unoccupied*V, f2);
  
  //shift by prior mean
  thrust::transform(prior_mean, prior_mean + num_unoccupied*V, thrust::plus<double>());
}