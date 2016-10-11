#ifndef ITERATOR_H
#define ITERATOR_H

#include<thrust/tuple.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/iterator/permutation_iterator.h>
#include<thrust/iterator/transform_iterator.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

//Used for generating rep(1:infinity, each=length) * incr
struct repEach: public thrust::unary_function<int, int>{
  int length;
  int incr;
  __host__ __device__ repEach(int length=1, int incr=1): length(length), incr(incr){}
  __host__ __device__ int operator()(int x){
    return (x/each)*incr; //integer division
  }
};

//Used for generating rep( (1:length)*incr, times=infinity)
struct repTimes: public thrust::unary_function<int, int>{
  int length;
  int incr;
  __host__ __device__ repTimes(int length=1, int incr=1): length(length), incr(incr){}
  __host__ __device__ int operator()(int x){
    return (x%length)*incr;
  }
};

//typedefs for custom fancy iterators (I)
typedef thrust::device_vector<double>::iterator realIter;
typedef thrust::device_vector<int>::iterator intIter;
typedef thrust::counting_iterator<int> countIter;
typedef thrust::transform_iterator<repTimes, countIter> repTimesIter;
typedef thrust::transform_iterator<repEach, countIter> repEachIter;


//Just gets an iterator it where it[i] = i
countIter getCountIter(){
  countIter cntIt = thrust::counting_iterator<int>(0);
  return cntIt;
}

//Gets an iterator for generating rep(1:length, times=infinity)
repTimesIter getRepTimesIter(int length, int incr, countIter countIt = getCountIter()){
  // repeat 0, rep, 2*rep, ..., length*rep ad nauseum
  repTimes f(length, incr);				 
  repTimesIter repIt = thrust::transform_iterator<repTimes, countIter>(countIt, f);
  return repIt;
}

//Gets an iterator for generating rep(1:infinity, each=each) * incr
repEachIter getRepEachIter(int length, int incr, countIter countIt = getCountIter()){
  // repeat each of i*incr, length times, i>=0
  repEach g(length, incr);
  repEachIter repIt = thrust::transform_iterator<repEach, countIter>(countIt, g);
  return repIt;
}


//For generating rep(arb_seq, times=infinity)
template<typename T>
struct gRepTimes{
  typedef thrust::permutation_iterator<T, repTimesIter> iterator;
};

//Gets an iterator for generating rep(arb_seq, times=infinity)
template<typename T>
typename gRepTimes<T>::iterator getGRepTimesIter(T begin, T end, int length, int incr=1, countIter countIt = getCountIter()){
  // repeats arbitrary vector, possibly strided
  repTimesIter cyc = getRepTimesIter(length, incr, countIt);
  typename gRepTimes<T>::iterator gRep = thrust::permutation_iterator<T, repTimesIter>(begin, cyc);
  return gRep;
}

//For generating rep(arb_seq, each= length)
template<typename T>
struct gRepEach{
  typedef thrust::permutation_iterator<T, repEachIter> iterator;
};

//Gets an iterator for generating rep(arb_seq, each= length)
template<typename T>
typename gRepEach<T>::iterator getGRepEachIter(T begin, T end, int length, int incr=1, countIter countIt = getCountIter()){
  // repeats each element along {0, incr, 2*incr, ...} length times
  repEachIter repeat = getRepEachIter(length, incr, countIt);
  typename gRepEach<T>::iterator gRep = thrust::permutation_iterator<T, repEachIter>(begin, repeat);
  return gRep;
}

// Belongs in a separate file
/*probably a better way to scope these defn's*/
typedef thrust::tuple<gRepTimes<realIter>::iterator, repEachIter> tup4RSReal;
typedef thrust::tuple<gRepTimes<intIter>::iterator, repEachIter>  tup4RSInt;
typedef thrust::zip_iterator<tup4RSReal> zip4RSReal;
typedef thrust::zip_iterator<tup4RSInt>  zip4RSInt;

//typedef and functor to perform tuple.first(double) + tuple.second(int)
typedef thrust::tuple<int, int> elt4RSInt;
struct sumZipRSInt : thrust::unary_function<elt4RSInt, double>{
  __host__ __device__ double operator()(elt4RSInt &tup){
    return (double)(thrust::get<0>(tup) + thrust::get<1>(tup));
  }
};

typedef thrust::transform_iterator<sumZipRSInt, zip4RSInt> RSIntIter;

// Use for creating key in reduce by key where what is needed are "row sums"
//Call function when you want to iterate over a key adding a constant increment each iteration
// "RS" = "repeated shifted"
RSIntIter getRSIntIter(intIter begin, intIter end, int incr, countIter countIt = getCountIter()){
  repEachIter eachIt = getRepEachIter(thrust::distance(begin, end), incr, countIt);
  gRepTimes<intIter>::iterator repIt = getGRepTimesIter(begin, end, thrust::distance(begin, end), 1, countIt);
  tup4RSInt tup = thrust::tuple<gRepTimes<intIter>::iterator, repEachIter>(repIt, eachIt);
  zip4RSInt zip = thrust::zip_iterator<tup4RSInt>(tup);
  sumZipRSInt f;
  RSIntIter result = thrust::transform_iterator<sumZipRSInt, zip4RSInt>(zip, f);
  return result;
}

// map from colmajor to rowmajor, i.e. transpose
struct colmaj_to_rowmaj{
  int R, C;
  __host__ __device__ colmaj_to_rowmaj(int R, int C): R(R), C(C){}
  __host__ __device__ int operator()(int i){
    return (i%C)*R + i/C;
  }
};

#endif
