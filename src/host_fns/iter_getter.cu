#include "../header/iter_getter.h"

//Just gets an iterator it where it[i] = i
countIter getCountIter(){
  countIter cntIt = thrust::counting_iterator<int>(0);
  return cntIt;
}

//Gets an iterator for generating rep(1:len, times=infinity)
repTimesIter getRepTimesIter(int len, int incr, countIter countIt){
  // repeat 0, incr, 2*incr, ..., len*incr ad nauseum
  repTimes f(len, incr);				 
  repTimesIter repIt = thrust::transform_iterator<repTimes, countIter>(countIt, f);
  return repIt;
}

//Gets an iterator for generating rep(1:infinity, each=each) * incr
repEachIter getRepEachIter(int len, int incr, countIter countIt){
  // repeat each of i*incr, len times, i>=0
  repEach g(len, incr);
  repEachIter repIt = thrust::transform_iterator<repEach, countIter>(countIt, g);
  return repIt;
}

rowIter getRowIter(int Rows, int row){
  // Row accessor, obj[iter + 1:C] returns obj[row, 1:C]
  countIter countIt = getCountIter();
  row_index f(Rows, row);	
  rowIter rowIt = thrust::transform_iterator<row_index, countIter>(countIt, f);
  return rowIt;
}

/* *******************
/* Use for creating key in reduce by key where what is needed are "row sums"
 * Call function when you want to iterate over a key adding a constant increment each iteration
 * "RS" = "repeated shifted"
 */
RSIntIter getRSIntIter(intIter begin, intIter end, int incr, countIter countIt){
  repEachIter eachIt = getRepEachIter(thrust::distance(begin, end), incr, countIt);
  gRepTimes<intIter>::iterator repIt = getGRepTimesIter(begin, end, thrust::distance(begin, end), 1, countIt);
  tup4RSInt tup = thrust::tuple<gRepTimes<intIter>::iterator, repEachIter>(repIt, eachIt);
  zip4RSInt zip = thrust::zip_iterator<tup4RSInt>(tup);
  sumZipRSInt f;
  RSIntIter result = thrust::transform_iterator<sumZipRSInt, zip4RSInt>(zip, f);
  return result;
}

/**********************************
 * This function gives an iterator to the transpose of
 * a flattened matrix stored on column-major format
 * 
 */
transposeIter getTransposeIter(int R, int C, countIter countIt){
  colmaj_to_rowmaj f(R, C);
  transposeIter t = thrust::transform_iterator<colmaj_to_rowmaj, countIter>(countIt, f);
  return t;
}


/**********************************************************8
 * Get an iterator to the diagonal elements of a matrix stored
 * in col-major format
 * 
 */
diagonalIter getDiagIter(int dim, countIter countIt){
  diag_elem f(dim);
  diagonalIter d = thrust::transform_iterator<diag_elem, countIter>(countIt, f);
  return d;
}


// Use for functions where only select columns are required
// "SC" = "select columns"
SCIntIter getSCIntIter(intIter begin, intIter end, int each, countIter countIt){

  repTimesIter timesIt = getRepTimesIter(each, 1, countIt);
  gRepEach<intIter>::iterator eachIt = getGRepEachIter(begin, end, each, 1, countIt);
  tup4SCInt tup = thrust::tuple<gRepEach<intIter>::iterator, repTimesIter>(eachIt, timesIt);
  zip4SCInt zip = thrust::zip_iterator<tup4SCInt>(tup);
  sumZipSCInt f(each);
  SCIntIter result = thrust::transform_iterator<sumZipSCInt, zip4SCInt>(zip, f);

  return result;
}
