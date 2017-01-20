#include "../header/iter_getter.h"
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include "../header/distribution.h"
#include "../header/printing.h"

int main(){

  using namespace thrust::placeholders;

  int rows = 6, cols = 10;
  
  fvec_d mat_d(rows*cols);
  
  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, rows*cols * sizeof(curandState)));
  
  getNormal<<<rows,cols>>>(devStates, thrust::raw_pointer_cast(mat_d.data()));

  fvec_h mat_h(rows*cols);
  thrust::copy(mat_d.begin(), mat_d.end(), mat_h.begin());
  
  ivec_h sel_cols(3);
  sel_cols[0] = 0;
  sel_cols[1] = 3;
  sel_cols[2] = 9;
  
  auto iter = thrust::make_permutation_iterator(mat_h.begin(),
    thrust::make_permutation_iterator(thrust::make_counting_iterator(0),
      thrust::make_transform_iterator(sel_cols.begin(), _1 * rows)));

  printVec(mat_h, rows, cols);
    
  thrust::copy(iter, iter + 3, std::ostream_iterator<double>(std::cout, " "));

  return 0;
}