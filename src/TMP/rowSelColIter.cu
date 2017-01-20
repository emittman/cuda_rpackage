#include "../header/iter_getter.h"
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include "../header/printing.h"
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
  
  auto iter = thrust::make_permutation_iterator(mat_h.begin(),
    thrust::make_permutation_iterator(thrust::make_counting_iterator(0),
      thrust::make_transform_iterator(sel_cols.begin(), _1 * rows)));

  printVec(mat_h, rows, cols);
    
  std::cout << "\nUsing auto:\n";
  thrust::copy(iter, iter + 3, std::ostream_iterator<double>(std::cout, " "));

  std::cout << "\nUsing explicit:\n";
  
  struct skip{
    int s;
    __host__ __device__ skip(int s): s(s){}
    __host__ __device__ int operator()(int i){
      return i*s;
    }
  };
  
  typename thrust::transform_iterator<skip, intIter> skipIter;
  typename thrust::permutation_iterator<realIter, skipIter> firstIter;
  
  skip f(rows);
  skipIter firstIndex = thrust::transform_iterator<skip, intIter>(sel_cols.begin(), f);
  firstIter firstElem = thrust::permutation_iterator<realIter, skipIter>(mat_h.begin(), firstIndex);

  thrust::copy(firstElem, firstElem + sel_cols.length(), std::ostream_iterator<double>(std::cout, " "));

  return 0;
}