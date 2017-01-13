#include "../header/cluster_probability.h"
#include "../header/iter_getter.h"
#include "../header/chain.h"
#include "cublas.h"

void big_matrix_multiply(fvec_d &A, fvec_d &B, fvec_d &big_grid, int a1, int a2, int b1, int b2){
  double alpha = 1, beta = 0;
  if(a1 != b1) std::cout << "a1 and b1 must be the same (t(A) B = big_grid)\n";
  int lda = a1, ldb = b1, ldc = a2;
  cublasHandle_t handle;
  cublasCreate(&handle);
  double *A_ptr = &(A[0]);
  double *B_ptr = &(B[0]);
  double *grid_ptr = &(big_grid[0]);
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, a2, b2, a1,
              &alpha, beta_ptr, lda,
              xty_ptr, ldb,
              &beta, grid_ptr, ldc);
  cublasDestroy(handle);
}
