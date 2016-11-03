#ifndef UTIL_CUDA_USAGE_H
#define UTIL_CUDA_USAGE_H

#include "include.h"


#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  REprintf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  REprintf("  %s\n", cudaGetErrorString(cudaGetLastError()));}}

int getDevice(){
	int device;
	cudaGetDevice(&device);
  return device;
}

int setDevice(int device) {
	int dev, deviceCount;
	cudaGetDeviceCount(&deviceCount);
  if(deviceCount < 1){
    Rprintf("No CUDA-capable GPUs detected.");
    return -1;
  }
  if(device < 0 || device >= deviceCount)
    Rprintf("Warning: invalid device index. Setting device = abs(device) mod deviceCount.\n");
  dev = abs(device) % deviceCount;
	CUDA_CALL(cudaSetDevice(dev));
  return dev;
}

#endif // UTIL_CUDA_USAGE_H
