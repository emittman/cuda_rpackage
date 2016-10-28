#ifndef SUMMARY_H
#define SUMMARY_H

#include "header/iter_getter.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

class summary{
public:
  int num_rows;
  int num_clusters;
  int num_columns;
  ivec_d clust_sums;
  ivec_d occupancy_count;
  ivec_d all;

 summary(int R, int Cl, int Co): num_rows(R), num_clusters(Cl), num_columns(Co), clust_sums(Cl*Co), occupancy_count(Cl), all(R*Co){
    //initialize with (silly) data
    thrust::sequence(all.begin(), all.end(), 0, 1);
}
  void update(ivec_d &key, int verbose);
};

#endif

// struct summary{
//   
//   int G;
//   int K;
//   int V;
//   ivec_d occupied;
//   ivec_d unoccupied;
//   int num_occupied;
//   ivec_d Mk;
//   fvec_d yty_sums;
//   fvec_d xty_sums;
//   fvec_d ytx_sums;
//   
//   summary(int _K, int _V, ivec_d zeta, data* dat);
//   void complete();
//   
// };
// 
// struct is_zero{
//   __host__ __device__ bool operator()(int i){
//     return i==0 ? 1 : 0;
//   }
// }
// 
// // zeta passed by value, data passed by reference
// summary::summary(int _K, int_V, ivec_d zeta, const data_t &data): G(_G), K(_K), V(_V), occupied(_K){
//   
//   // local allocations
//   ivec_d perm(G); // will store permutation
//   thrust::sequence(perm.begin(), perm.end(), 0, 1);
//   
//   // sort, identify occupied
//   thrust::sort_by_key(zeta.begin(), zeta.end(), perm.begin());
//   ivec_d::iterator endptr;
//   endptr = thrust::unique_copy(zeta.begin(), zeta.end(), occupied.begin());
//   occupied.erase(endptr, occupied.end());
//   num_occupied = occupied.size();
//   
//   // calculate Mk
//   Mk(K, 0); //call constructor
//   thrust::permutation_iterator<intIter, intIter> mapMk = thrust::permutation_iterator<intIter,intIter>(M_k.begin(), occupied.begin());
//   thrust::reduce_by_key(zeta.begin(), zeta.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), mapMk);
//   
//   // unoccupied
//   unoccupied.reserve(K - num_occupied);
//   endptr = thrust::copy_if(thrust::make_counting_iterator<int>(0),
//                            thrust::make_counting_iterator<int>(K), Mk.begin(), unoccupied.begin(), is_zero())
//     unoccupied.erase(endptr, unoccupied.end());
//   
//   //size vectors
//   yty_sums.reserve(num_occupied);
//   xty_sums.reserve(num_occupied*V);
//   ytx_sums.reserve(num_occupied*V);
//   
//   //yty
//   thrust::reduce_by_key(zeta.begin(), zeta.end(), data.yty.begin(), thrust::make_discard_iterator(), yty.sums.begin());
//   
//   /* ytx_sums
//   * 
//   */
//   // broadcast key
//   RSIntIter zeta_rep = getRSIntIter(zeta.begin(), zeta.end(), K);
//   // "arrange" data
//   RSIntIter in_index = getRSIntIter(perm.begin(),perm.end(), G);
//   permutation_iterator<realIter, intIter> sort_ytx = permutation_iterator<realIter, intIter>(ytx.begin(), in_index);
//   // reduce
//   reduce_by_key(zeta_rep, zeta_rep + G*V, sort_ytx, hrust::make_discard_iterator(), ytx_sums.begin());
//   
//   /* xty_sums
//   * 
//   */
//   transpose(ytx_sums.begin(), ytx_sums.end(), xty_sums.begin(), num_occupied, V);
//   
// }