#ifndef ITER_GETTER_H
#define ITER_GETTER_H

#include "iterator.h"

//Just gets an iterator it where it[i] = i
countIter getCountIter();

//Gets an iterator for generating rep(1:len, times=infinity)
repTimesIter getRepTimesIter(int len, int incr, countIter countIt = getCountIter());
//Gets an iterator for generating rep(arb_seq, times=infinity)
template<typename T>
typename gRepTimes<T>::iterator getGRepTimesIter(const T &begin, const T &end, int len, int incr=1, countIter countIt = getCountIter()){
  // repeats arbitrary vector, possibly strided
  repTimesIter cyc = getRepTimesIter(len, incr, countIt);
  typename gRepTimes<T>::iterator gRep = thrust::permutation_iterator<T, repTimesIter>(begin, cyc);
  return gRep;
}


//Gets an iterator for generating rep(1:infinity, each=each) * incr
repEachIter getRepEachIter(int len, int incr, countIter countIt = getCountIter());
//Gets an iterator for generating rep(arb_seq, each= len)
template<typename T>
typename gRepEach<T>::iterator getGRepEachIter(T begin, T end, int len, int incr=1, countIter countIt = getCountIter()){
  // repeats each element along {0, incr, 2*incr, ...} len times
  repEachIter repeat = getRepEachIter(len, incr, countIt);
  typename gRepEach<T>::iterator gRep = thrust::permutation_iterator<T, repEachIter>(begin, repeat);
  return gRep;
}

//Gets an iterator for accessing rows of a matrix
rowIter getRowIter(int Rows, int row);


// Use for creating key in reduce by key where what is needed are "row sums"
//Call function when you want to iterate over a key adding a constant increment each iteration
// "RS" = "repeated shifted"
RSIntIter getRSIntIter(intIter begin, intIter end, int incr, countIter countIt = getCountIter());

/**********************************
 * This function gives an iterator to the transpose of
 * a flattened matrix stored on column-major format
 * 
 */
transposeIter getTransposeIter(int R, int C, countIter countIt = getCountIter());

template<typename T>
typename gTranspose<T>::iterator getGTransposeIter(const T &begin, const T &end, int R, int C, countIter countIt = getCountIter()){
  transposeIter trans = getTransposeIter(R, C, countIt);
  typename gTranspose<T>::iterator gTrans = thrust::permutation_iterator<T, transposeIter>(begin, trans);
  return gTrans;
}

/**********************************************************8
 * Get an iterator to the diagonal elements of a matrix stored
 * in col-major format
 * 
 */
diagonalIter getDiagIter(int dim, countIter countIt = getCountIter());
template<typename T>
typename gDiagonal<T>::iterator getGDiagIter(T begin, T end, int dim, countIter countIt = getCountIter()){
  diagonalIter diag = getDiagIter(dim, countIt);
  typename gDiagonal<T>::iterator gDiag = thrust::permutation_iterator<T, diagonalIter>(begin, diag);
  return gDiag;
}

// Use for functions where only select columns are required
// "SC" = "select columns"
SCIntIter getSCIntIter(intIter begin, intIter end, int each, countIter countIt = getCountIter());

#endif