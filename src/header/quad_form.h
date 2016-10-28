#ifndef QUAD_FORM_H
#define QUAD_FORM_H

#include "iter_getter.h"


//Compute t(x_i) %*% A %*% x_i where i=0, ..., n-1
void quad_form_multi(fvec_d &A, fvec_d &x, fvec_d &y, int n, int dim);


#endif
