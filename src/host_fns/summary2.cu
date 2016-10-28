#include "../header/iter_getter.h"
#include "../header/summary2.h"
#include "../header/chain.h"
#include "../header/transpose.h"
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

struct is_zero: thrust::unary_function<int, bool>{
  __host__ __device__ bool operator()(int i){
    return i==0 ? 1 : 0;
  }
};

// zeta passed by value, data passed by reference
summary2::summary2(int _G, int _K, int _V, ivec_d zeta, const data_t &data): G(_G), K(_K), V(_V), occupied(_K), Mk(_K, 0){
  
  // local allocations
  ivec_d perm(G); // will store permutation
  thrust::sequence(perm.begin(), perm.end(), 0, 1);
  
  // sort, identify occupied
  thrust::sort_by_key(zeta.begin(), zeta.end(), perm.begin());
  ivec_d::iterator endptr;
  endptr = thrust::unique_copy(zeta.begin(), zeta.end(), occupied.begin());
  occupied.erase(endptr, occupied.end());
  num_occupied = occupied.size();
  
  // calculate Mk
  thrust::permutation_iterator<intIter, intIter> mapMk = thrust::permutation_iterator<intIter,intIter>(Mk.begin(), occupied.begin());
  thrust::reduce_by_key(zeta.begin(), zeta.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), mapMk);
  
  // unoccupied
  unoccupied.reserve(K - num_occupied);
  endptr = thrust::copy_if(thrust::make_counting_iterator<int>(0),
                           thrust::make_counting_iterator<int>(K), Mk.begin(), unoccupied.begin(), is_zero());
  unoccupied.erase(endptr, unoccupied.end());
  
  //size vectors
  yty_sums.reserve(num_occupied);
  xty_sums.reserve(num_occupied*V);
  ytx_sums.reserve(num_occupied*V);
  
  /*yty_sums
   *
   */
  thrust::reduce_by_key(zeta.begin(), zeta.end(), data.yty.begin(), thrust::make_discard_iterator(), yty_sums.begin());
  
  /* ytx_sums
   * 
   */
  // broadcast key
  RSIntIter zeta_rep = getRSIntIter(zeta.begin(), zeta.end(), K);
  // "arrange" data
  RSIntIter in_index = getRSIntIter(perm.begin(),perm.end(), G);
  thrust::permutation_iterator<realIter, RSIntIter> sort_ytx = thrust::permutation_iterator<realIter, RSIntIter>(data.ytx.begin(), in_index);
  // reduce
  thrust::reduce_by_key(zeta_rep, zeta_rep + G*V, sort_ytx, thrust::make_discard_iterator(), ytx_sums.begin());
  
  /* xty_sums
  * 
    */
    transpose<realIter>(ytx_sums.begin(), ytx_sums.end(), num_occupied, V, xty_sums.begin());
  
}