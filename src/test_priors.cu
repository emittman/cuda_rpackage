#include "chain.h"
#include <iostream>
#include "printing.h"

int main(){
  
  int V = 2;
  double alpha = 1.0;
  double a = 2.0;
  double b = 3.0;
  
  fvec_h mu0(V);
  fvec_h lambda2(V);
  mu0[0] = 3.0;
  mu0[1] = 30.0;
  double *mu_ptr = &(mu0[0]);
  lambda2[0] = 2.0;
  lambda2[1] = 3.0;
  double *lm_ptr = &(lambda2[0]);
  priors_t prior(V, mu_ptr, lm_ptr, alpha, a, b);
  std::cout << "I made a prior!\n mu0:\n";
  printVec(prior.mu0, V, 1);
  
  
  

  return 0;
}
