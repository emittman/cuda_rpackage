#ifndef BETA_HAT
#define BETA_HAT

#include "iter_getter.h"
#include "cublas_v2.h"
#include "printing.h"
#include "thrust/device_vector.h"
#include "thrust/for_each.h"

void beta_hat(fvec_d &chol_prec, fvec_d &beta_hat, int K_occ, int V);
void scale_chol_inv(fvec_d &chol_prec, fvec_d &z, int n, int dim);

#endif