struct priors_t{
  
  int V;
  fvec_d mu0;
  fvec_d lambda2;
  double alpha;
  double a;
  double b;
  
  priors_t(int _V, double* _mu0, double* _lambda2, double _alpha, double _a, double _b) : V(_V), alpha(_alpha), a(_a), b(_b){
    mu0 = fvec_d(_mu0, _mu0 + V);
    lambda2 = fvec_d(_lambda2, _lambda2 + V);
  }

};

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
