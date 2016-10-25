#ifndef SUMMARY_FN_H
#define SUMMARY_FN_H

#include "summary.h"
#include "iterator2.h"
#include "printing.h"
#include<thrust/tuple.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/iterator/permutation_iterator.h>
#include<thrust/iterator/transform_iterator.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/unique.h>
#include<thrust/distance.h>
#include<thrust/iterator/constant_iterator.h>
#include<thrust/sort.h>
#include<thrust/reduce.h>


void summary::update(ivec_d &key, int verbose=0){
 // check length of cluster index
  if(key.size() != num_rows){
    std::cout << "Key is wrong size" << std::endl;
    return;
  }

  // local allocations
  ivec_d perm(num_rows); // will store permutation
  thrust::sequence(perm.begin(), perm.end(), 0, 1);
  ivec_d unique_key(num_clusters);

  // sort the key, capturing the permutation to do so in perm
  thrust::sort_by_key(key.begin(), key.end(), perm.begin());

  // identify occupied clusters
  ivec_d::iterator last_occ;

  //get unique values of occupied clusters and capture the location of the last one
  last_occ = thrust::unique_copy(key.begin(), key.end(), unique_key.begin());

  // resize to ensure that unique_key.size() = number of occupied clusters
  unique_key.erase(last_occ, unique_key.end());

  //allocation for debug
  ivec_d tmp(num_rows * num_columns);

  // tabulate the cluster assignments
  thrust::reduce_by_key(key.begin(), key.end(), thrust::constant_iterator<int>(1), thrust::make_discard_iterator(),
			thrust::make_permutation_iterator(occupancy_count.begin(), unique_key.begin()));

  //debug
  if(verbose>1){
    std::cout << "num occupied:\n" << unique_key.size() << std::endl;
    std::cout << "occupancy_count:" << std::endl;
    printVec(occupancy_count, num_clusters, 1);
    std::cout << "(sorted) key:\n";
    printVec(key, num_rows, 1);
    std::cout << "unsorted to reduce:\n";
    printVec(all, num_rows, num_columns);
  }

  // maps i = {0, 1, 2, ...} to all[perm[i%perm.size()] + i/num_rows * num_rows]
  RSIntIter map_obs = getRSIntIter(perm.begin(), perm.end(), num_rows);

  //debug
  if(verbose > 1){
    std::cout << "sorted to reduce:\n";
    thrust::copy(map_obs, map_obs + num_rows * num_columns, tmp.begin());
    printVec(tmp, num_rows, num_columns);
  }
  
  // maps i = {0, 1, 2, ...} to key[i%key.size()] + i/key.size() * num_clusters 
  RSIntIter brdcst_key = getRSIntIter(key.begin(), key.end(), num_clusters);

  //debug
  if(verbose > 1){
    std::cout << "reducing by this key:\n";
    thrust::copy(brdcst_key, brdcst_key + num_rows * num_columns, tmp.begin());
    printVec(tmp, num_rows, num_columns);
    RSIntIter pdgnhl_sums = getRSIntIter(unique_key.begin(), unique_key.end(), num_clusters);
    std::cout << "pidgeonhole results:\n";
    thrust::copy(pdgnhl_sums, pdgnhl_sums + num_clusters * num_columns, tmp.begin());
    printVec(tmp, num_clusters, num_columns);
  }

  // maps i = {0, 1, 2, ...} to clust_sums[unique_key[i%unique_key.size()] + i/num_clusters * num_clusters]
  typedef thrust::permutation_iterator<intIter, RSIntIter> trnsByStdCyIter;
  RSIntIter map0 = getRSIntIter(unique_key.begin(), unique_key.end(), num_clusters);
  trnsByStdCyIter scatter_sums = thrust::permutation_iterator<intIter, RSIntIter>(clust_sums.begin(), map0);

  
  // sum rows of all by key and store in appropriate row in clust_sums
  reduce_by_key(brdcst_key, brdcst_key + num_rows*num_columns, map_obs,
		thrust::make_discard_iterator(),
		scatter_sums);

  //debug
  if(verbose > 1){
    std::cout << "result of reduction:\n";
    printVec(clust_sums, num_clusters, num_columns);
  }

}

#endif
