#ifndef ITERATOR_H
#define ITERATOR_H

#include<thrust/tuple.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/iterator/permutation_iterator.h>
#include<thrust/iterator/transform_iterator.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/device_ptr.h>

//general purpose typedefs
typedef thrust::device_vector<int> ivec_d;
typedef thrust::device_vector<double> fvec_d;
typedef thrust::device_vector<int>::iterator intIter;
typedef thrust::device_vector<double>::iterator realIter;
typedef thrust::host_vector<int> ivec_h;
typedef thrust::host_vector<double> fvec_h;


/****************************************
* Iterators for repeating consecutive indices simliar to rep(1:N) in R
* repEach and repTimes
*/

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

//typedefs
typedef thrust::counting_iterator<int> countIter;
typedef thrust::transform_iterator<repTimes, countIter> repTimesIter;
typedef thrust::transform_iterator<repEach, countIter> repEachIter;


/*************
* Generalize repEach/repTimes; similar to rep(X) in R
*/

//For generating rep(arb_seq, times=infinity)
template<typename T>
struct gRepTimes{
  typedef thrust::permutation_iterator<T, repTimesIter> iterator;
};

//For generating rep(arb_seq, each= len)
template<typename T>
struct gRepEach{
  typedef thrust::permutation_iterator<T, repEachIter> iterator;
};


/************
 *  Row indices and row accessor (for colmajor matrix)
 *  R = number of rows
 *  r = given row
 */
struct row_index: public thrust::unary_function<int, int>{
  int R, r;
  __host__ __device__ row_index(int _R, int _r): R(_R), r(_r){}
  __host__ __device__ int operator()(int &x){
    return R*x + r;
  }
};

typedef thrust::transform_iterator<row_index, countIter> rowIter;
typedef thrust::permutation_iterator<fvec_d::iterator, rowIter> strideIter;



/*************************
 * The main function is getRSIntIter which is equivalent to
 * rep(X, times) + rep(1:cols, each=colsize) * colsize
 * 
 */
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



/*********************
 * Transpose iterator
 * maps from colmajor to rowmajor, i.e. A[i + m*j] to A[j + n*i]
 * 
 */
struct colmaj_to_rowmaj : thrust::unary_function<int,int>{
  int R, C;
  __host__ __device__ colmaj_to_rowmaj(int R, int C): R(R), C(C){}
  __host__ __device__ int operator()(int i){
    return (i%C)*R + i/C;
  }
};

// definition for transpose iterators
typedef thrust::transform_iterator<colmaj_to_rowmaj, countIter> transposeIter;

template <typename T>
struct gTranspose{
  typedef thrust::permutation_iterator<T, transposeIter> iterator;
};

/**********************************************************8
 * Get an iterator to the diagonal elements of a matrix stored
 * in col-major format
 * 
 */


// def and getter for diagonalIter
struct diag_elem: thrust::unary_function<int,int>{
  int dim;

  diag_elem(int dim): dim(dim){}
  
  __host__ __device__ int operator()(int i){
    return (i%dim) * (dim+1) + (i/dim)*dim*dim;
  }
};

typedef thrust::transform_iterator<diag_elem, countIter> diagonalIter;


template<typename T>
struct gDiagonal{
  typedef thrust::permutation_iterator<T, diagonalIter> iterator;
};


/*********************************************
 * An iterator for accessing elements of a matrix
 * given a vector of column indices
 * 
 */

//def and getter for (SelectColumns) SCIntIter
typedef thrust::tuple<gRepEach<realIter>::iterator, repTimesIter> tup4SCReal;
typedef thrust::tuple<gRepEach<intIter>::iterator, repTimesIter>  tup4SCInt;
typedef thrust::zip_iterator<tup4SCReal> zip4SCReal;
typedef thrust::zip_iterator<tup4SCInt>  zip4SCInt;

//typedef and functor to perform tuple.first(double) + tuple.second(int)
typedef thrust::tuple<int, int> elt4SCInt;

struct sumZipSCInt : thrust::unary_function<elt4SCInt, int>{
  int stride;
  __host__ __device__ sumZipSCInt(int s): stride(s){}

  __host__ __device__ double operator()(elt4SCInt &tup){
    return thrust::get<0>(tup)*stride + thrust::get<1>(tup);
  }

};

typedef thrust::transform_iterator<sumZipSCInt, zip4SCInt> SCIntIter;


#endif
