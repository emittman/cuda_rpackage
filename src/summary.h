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

// class summary{
// public:
//   int G;
//   int K;
//   int V;
//   ivec_d occupied;
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
// summary::summary(int _K, int_V, ivec_d zeta, const data_t &dat): G(_G), K(_K), V(_V), occupied(_K){
//   
//   // local allocations
//   ivec_d perm(G); // will store permutation
//   thrust::sequence(perm.begin(), perm.end(), 0, 1);
// 
//   // sort the key, capturing the permutation to do so in perm
//   thrust::sort_by_key(zeta.begin(), zeta.end(), perm.begin());
//   
//   // identify occupied clusters
//   ivec_d::iterator last_occ;
//   
//   //get unique values of occupied clusters and capture the location of the last one
//   last_occ = thrust::unique_copy(zeta.begin(), zeta.end(), occupied.begin());
//   
//   // resize to ensure that unique_key.size() = number of occupied clusters
//   occupied.erase(last_occ, occupied.end());
//   num_occupied = occupied.size();
//   
//   //size vectors
//   Mk.reserve(num_occupied);
//   yty_sums.reserve(num_occupied);
//   xty_sums.reserve(num_occupied*V);
//   ytx_sums.reserve(num_occupied*V);
//   
//   // ...
// }