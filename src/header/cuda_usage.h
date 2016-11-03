#ifndef UTIL_CUDA_USAGE_H
#define UTIL_CUDA_USAGE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  REprintf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  REprintf("  %s\n", cudaGetErrorString(cudaGetLastError()));}}

int getDevice();

int setDevice(int device);

#endif // UTIL_CUDA_USAGE_H
