#ifndef DOT_PROD
#define DOT_PROD

#include "iter_getter.h"
#include "cublas_v2.h"
#include "printing.h"
#include "thrust/device_vector.h"
#include "thrust/for_each.h"

void multi_dot_prod(fvec_d &x, fvec_d &y, fvec_d &z, int dim, int n);

#endif