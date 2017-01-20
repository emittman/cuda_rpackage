#include "../header/iter_getter.h"
#include <thrust/sequence.h>
#include <thrust/functional.h>

int main(){

  using namespace thrust::placeholders;

  int rows = 6, cols = 10;
  
  fvec_h mat_h(rows*cols);
  thrust::sequence(mat_h.begin(), mat_h.end());
  
  ivec_h sel_cols(3);
  sel_cols[0] = 0;
  sel_cols[1] = 3;
  sel_cols[2] = 9;
  
  auto iter = thrust::make_permutation_iterator(mat_h.begin(), thrust::make_transform_iterator(sel_cols.begin(), _1 * rows));
  
  thrust::copy(iter, iter + 3, std::ostream_iterator<double>(std::cout, " "));

  return 0;
}