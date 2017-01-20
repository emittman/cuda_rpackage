#include <thrust/sequence.h>
#include <thrust/functional.h>
#include "../header/printing.h"
#include "../header/iter_getter.h"
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

int main(){

  using namespace thrust::placeholders;

  int rows = 6, cols = 10;
  
  // create a minstd_rand object to act as our source of randomness
  thrust::minstd_rand rng;
  // create a uniform_real_distribution to produce floats from [-7,13)
  thrust::uniform_real_distribution<float> dist(-3,3);
  // write a random number from the range [-7,13) to standard output
  
  
  fvec_h mat_h(rows*cols);
  for(int i=0; i<rows*cols; ++i) mat_h[i] = dist(rng);
  
  ivec_h sel_cols(3);
  sel_cols[0] = 0;
  sel_cols[1] = 3;
  sel_cols[2] = 9;
  
  ivec_d selCols(sel_cols);
  fvec_d mat_d(mat_h);
  
  std::cout << "Matrix:\n";
  printVec(mat_h, rows, cols);
  
  std::cout << "\nUsing explicit iterator types:\n";
  
  std::cout << "\n to capture first element in columns:\n";
  printVec(sel_cols, 3, 1);
  
  gSFRIter<realIter>::iterator firstElem = getGSFRIter(mat_d.begin(), mat_d.end(), selCols, rows);
  
  thrust::copy(firstElem, firstElem + selCols.size(), std::ostream_iterator<double>(std::cout, " "));

  return 0;
}