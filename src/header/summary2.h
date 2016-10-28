#ifndef SUMMARY2_H
#define SUMMARY2_H

#include "iterator.h"
#include "chain.h"
#include "printing.h"

struct summary2{
  
  int G;
  int K;
  int V;
  ivec_d occupied;
  ivec_d unoccupied;
  int num_occupied;
  ivec_d Mk;
  fvec_d yty_sums;
  fvec_d xty_sums;
  fvec_d ytx_sums;
  
  summary2(int _G, int _K, int _V, ivec_d zeta, data_t &dat);
  void print_Mk(){ printVec(Mk, 1, K);}
  void print_yty(){ printVec(yty_sums, 1, num_occupied);}
  void print_xty(){ printVec(xty_sums, V, num_occupied);}
};

#endif