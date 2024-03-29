#ifndef DISTR_H
#define DISTR_H

#include "curand_kernel.h"
#include "cuda_usage.h"
#include <R.h>

__device__ double rgamma(curandState *state, double a, double b, bool logscale);

__device__ double rgamma2(curandState *state, double a, double b, bool logscale);

__device__ double rbeta(curandState *state,  double a, double b, bool logscale);

__global__ void setup_kernel(int seed, int n_threads, curandState *states);

__global__ void getGamma(curandState *states, int n_threads, double *a, double *b, double *result, bool logscale);

__global__ void getBeta(curandState *states, int n_threads, double *a, double *b, double *result, bool logscale);

__global__ void getUniform(curandState *states, int n_threads, double *upper_result);
                          
__global__ void getExponential(curandState *states, int n_threads, double *weights, double *result);

__global__ void getNormal(curandState *states, int n_threads, double *result);

#endif
