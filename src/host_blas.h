#ifndef BLAS_H
#define BLAS_H
//#include <R_ext/BLAS.h>
// extern void daxpy(int*, double*, double*, int*, double*, int*); 
#include "iterator2.h"

extern "C"{
  double dgemm_(char* transa, char* transb, int m, int n, int k,
                double alpha, double *A, int lda, double *B, int ldb,
                double beta, double *C, int ldc);
}

void tself_self_multiply(int rows, int cols, double *x, double *result){
  fvec_h tmp(m*n);
  char op1 = 'T', op2 = 'N';
  int lda = rows, ldb = rows, ldc = cols;
  dgemm_(&op1, &op2, cols, cols, rows, 1.0, x, lda, x, ldb, 0.0, result, cols);
}

#endif
  