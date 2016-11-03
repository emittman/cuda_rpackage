#ifndef DISTR_H
#define DISTR_H

#include "curand_kernel.h"
#include "thrust/functional.h"
#include "util/cuda_usage.h"

__device__ double rgamma(curandState *state, double a, double b);

__device__ double rbeta(curandState *state,  double a, double b);

__global__ void setup_kernel(curandState *states);

__global__ void getBeta(curandState *states, double *a, double *b, double *result);

#endif