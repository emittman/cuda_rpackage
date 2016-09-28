#ifndef ITERATOR_H
#define ITERATOR_H

#include<thrust/tuple.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/iterator/permutation_iterator.h>
#include<thrust/iterator/transform_iterator.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/host_vector.h>

//functors for custom fancy iterators

struct repOverCols: public thrust::unary_function<int, int>{
  int num_rows;
  int incr;
  __host__ __device__ repOverCols(int R, int incr): num_rows(R), incr(incr){}
  __host__ __device__ int operator()(int x){
    return (x/num_rows)*incr; //integer division; floor(linear_index/num_rows) * jump
  }
};

struct repWithinCols: public thrust::unary_function<int, int>{
  int num_rows;
  __host__ __device__ repWithinCols(int R): num_rows(R){}
  __host__ __device__ int operator()(int x){
    return x%num_rows;
  }
};

//typedefs for custom fancy iterators

typedef thrust::host_vector<int>::iterator intIter;
typedef thrust::counting_iterator<int> countIter;
typedef thrust::transform_iterator<repWithinCols, countIter> cycleIter;
typedef thrust::transform_iterator<repOverCols, countIter> columnIter;
typedef thrust::permutation_iterator<intIter, cycleIter> gnCycleIter;
typedef thrust::tuple<columnIter,gnCycleIter> tup4stridedCycle;
typedef thrust::zip_iterator<tup4stridedCycle> zip4stridedCycleIter;

countIter getCountIter(){
  countIter cntIt = thrust::counting_iterator<int>(0);
  return cntIt;
}

cycleIter getCycleIter(int length){
  // repeating sequence of 0, ..., length
  countIter countIt = getCountIter();
  repWithinCols f(length);				 
  cycleIter cycIt = thrust::transform_iterator<repWithinCols, countIter>(countIt, f);
  return cycIt;
}

columnIter getColumnIter(int length, int incr){
  // repeat each nonnegative int "length" times, jumping up by length if incr=true
  countIter countIt = getCountIter();
  repOverCols g(length, incr);
  columnIter colIt = thrust::transform_iterator<repOverCols, countIter>(countIt, g);
  return colIt;
}

gnCycleIter getGnrlCycleIter(thrust::host_vector<int> &perm){
  // repeats arbitrary finite sequence
  cycleIter cyc = getCycleIter(perm.size());
  gnCycleIter prmIt = thrust::permutation_iterator<intIter, cycleIter>(perm.begin(), cyc);
  return prmIt;
}

/* These take an existing countIter as an argument */
cycleIter getCycleIter2(countIter countIt, int length){
  repWithinCols f(length);				 
  cycleIter cycIt = thrust::transform_iterator<repWithinCols, countIter>(countIt, f);
  return cycIt;
}

columnIter getColumnIter2(countIter countIt, int length, int incr){
  repOverCols f(length, incr);
  columnIter colIt = thrust::transform_iterator<repOverCols, countIter>(countIt, f);
  return colIt;
}

gnCycleIter getGnrlCycleIter2(countIter countIt, thrust::host_vector<int> &perm){
  // repeats arbitrary finite sequence
  cycleIter cyc = getCycleIter2(countIt, perm.size());
  gnCycleIter prmIt = thrust::permutation_iterator<intIter, cycleIter>(perm.begin(), cyc);
  return prmIt;
}

// functor for getStridedCycleIter()
struct sumZip4strided: public thrust::unary_function<tup4stridedCycle, int>
{
  template<typename Tuple>
  __host__ __device__
  int operator()(const Tuple& tup){
    return (thrust::get<0>(tup) + thrust::get<1>(tup));
  }
};
//return type iterator for getStridedCycleIter()
//depends on sumZip4strided
typedef thrust::transform_iterator<sumZip4strided, zip4stridedCycleIter> stridedCycleIter;

stridedCycleIter getStridedCycleIter(thrust::host_vector<int> &perm, int incr){
  // combines getGnrlCycleIter with getColumnIter
  countIter cntIt = getCountIter();
  columnIter colIt = getColumnIter2(cntIt, perm.size(), incr);
  gnCycleIter gncycIt = getGnrlCycleIter2(cntIt, perm);
  tup4stridedCycle tup(colIt, gncycIt);
  zip4stridedCycleIter zip(tup);
  sumZip4strided f;
  stridedCycleIter iter = thrust::transform_iterator<sumZip4strided, zip4stridedCycleIter>(zip, f);
  return iter;
}

typedef thrust::permutation_iterator<intIter, stridedCycleIter> trnsByStdCyIter;

trnsByStdCyIter getTrnsByStdCyIter(thrust::host_vector<int> &perm, thrust::host_vector<int> &original, int incr){
  stridedCycleIter permby = getStridedCycleIter(perm, incr);
  trnsByStdCyIter trans_iter = thrust::permutation_iterator<intIter, stridedCycleIter>(original.begin(), permby);
  return trans_iter;
}
  
#endif
