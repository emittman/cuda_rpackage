#ifndef ITERATOR2_H
#define ITERATOR2_H

#include<thrust/tuple.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/iterator/permutation_iterator.h>
#include<thrust/iterator/transform_iterator.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/device_ptr.h>

//Used for generating rep(1:infinity, each=len) * incr
struct repEach: public thrust::unary_function<int, int>{
  int len;
  int incr;
  __host__ __device__ repEach(int len=1, int incr=1): len(len), incr(incr){}
  __host__ __device__ int operator()(int x){
    return (x/len)*incr; //integer division
  }
};

//Used for generating rep( (1:len)*incr, times=infinity)
struct repTimes: public thrust::unary_function<int, int>{
  int len;
  int incr;
  __host__ __device__ repTimes(int len=1, int incr=1): len(len), incr(incr){}
  __host__ __device__ int operator()(int x){
    return (x%len)*incr;
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

//Gets an iterator for generating rep(1:len, times=infinity)
repTimesIter getRepTimesIter(int len, int incr, countIter countIt = getCountIter()){
  // repeat 0, incr, 2*incr, ..., len*incr ad nauseum
  repTimes f(len, incr);				 
  repTimesIter repIt = thrust::transform_iterator<repTimes, countIter>(countIt, f);
  return repIt;
}

//Gets an iterator for generating rep(1:infinity, each=each) * incr
repEachIter getRepEachIter(int len, int incr, countIter countIt = getCountIter()){
  // repeat each of i*incr, len times, i>=0
  repEach g(len, incr);
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
typename gRepTimes<T>::iterator getGRepTimesIter(const T &begin, const T &end, int len, int incr=1, countIter countIt = getCountIter()){
  // repeats arbitrary vector, possibly strided
  repTimesIter cyc = getRepTimesIter(len, incr, countIt);
  typename gRepTimes<T>::iterator gRep = thrust::permutation_iterator<T, repTimesIter>(begin, cyc);
  return gRep;
}

//For generating rep(arb_seq, each= len)
template<typename T>
struct gRepEach{
  typedef thrust::permutation_iterator<T, repEachIter> iterator;
};

//Gets an iterator for generating rep(arb_seq, each= len)
template<typename T>
typename gRepEach<T>::iterator getGRepEachIter(T begin, T end, int len, int incr=1, countIter countIt = getCountIter()){
  // repeats each element along {0, incr, 2*incr, ...} len times
  repEachIter repeat = getRepEachIter(len, incr, countIt);
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
struct colmaj_to_rowmaj : thrust::unary_function<int,int>{
  int R, C;
  __host__ __device__ colmaj_to_rowmaj(int R, int C): R(R), C(C){}
  __host__ __device__ int operator()(int i){
    return (i%C)*R + i/C;
  }
};

// definition and getter for transposeIter
typedef thrust::transform_iterator<colmaj_to_rowmaj, countIter> transposeIter;

transposeIter getTransposeIter(int R, int C, countIter countIt = getCountIter()){
  colmaj_to_rowmaj f(R, C);
  transposeIter t = thrust::transform_iterator<colmaj_to_rowmaj, countIter>(countIt, f);
  return t;
}

template <typename T>
struct gTranspose{
  typedef thrust::permutation_iterator<T, transposeIter> iterator;
};

template<typename T>
typename gTranspose<T>::iterator getGTransposeIter(const T &begin, const T &end, int R, int C, countIter countIt = getCountIter()){
  transposeIter trans = getTransposeIter(R, C, countIt);
  typename gTranspose<T>::iterator gTrans = thrust::permutation_iterator<T, transposeIter>(begin, trans);
  return gTrans;
}

struct diag_elem: thrust::unary_function<int,int>{
  int dim;

  diag_elem(int dim): dim(dim){}
  
  __host__ __device__ int operator()(int i){
    return (i%dim) * (dim+1) + (i/dim)*dim*dim;
  }
};

typedef thrust::transform_iterator<diag_elem, countIter> diagonalIter;

diagonalIter getDiagIter(int dim, countIter countIt = getCountIter()){
  diag_elem f(dim);
  diagonalIter d = thrust::transform_iterator<diag_elem, countIter>(countIt, f);
  return d;
}

template<typename T>
struct gDiagonal{
  typedef thrust::permutation_iterator<T, diagonalIter> iterator;
};

template<typename T>
typename gDiagonal<T>::iterator getGDiagIter(T begin, T end, int dim, countIter countIt = getCountIter()){
  diagonalIter diag = getDiagIter(dim, countIt);
  typename gDiagonal<T>::iterator gDiag = thrust::permutation_iterator<T, diagonalIter>(begin, diag);
  return gDiag;
}



#endif
