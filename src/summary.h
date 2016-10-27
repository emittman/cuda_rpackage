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

/*
typedef thrust::permutation_iterator<intIter, cycleIter> permuteRowsIter;
typedef thrust::tuple<columnIter, permuteRowsIter> iterTuple;
typedef thrust::zip_iterator<iterTuple> zipIter;

// functor for .update()
struct sumZip: public thrust::unary_function<iterTuple, int>
{
  template<typename Tuple>
  __host__ __device__
  int operator()(const Tuple& tup){
    return (thrust::get<0>(tup) + thrust::get<1>(tup));
  }
};

typedef thrust::transform_iterator<sumZip, zipIter> outIter;
typedef thrust::permutation_iterator<intIter, outIter> alignMatrixIter;
*/

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
