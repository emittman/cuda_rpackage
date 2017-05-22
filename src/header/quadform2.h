#ifndef QUADFORM2
#define QUADFORM2

#include "iterator.h"

void quadform_multipleK(fvec_d &beta, fvec_d &xtx, fvec &result, int K, int V);

void quadform_multipleGK(fvec_d &beta, fvec_d &xtx, fvec &result, int G, int K, int V);

void quadform_multipleMatch(fvec_d &beta, fvec_d &xtx, fvec &result, int K, int V);

#endif