#ifndef DISTR_H
#define DISTR_H

#include "curand_kernel.h"
#include "thrust/functional.h"
#include "util/cuda_usage.h"
  
  __device__ double rgamma (curandState *state, const double a, const double b){
    /* assume a > 0 */
      
      if (a < 1){
        double u = curand_uniform_double(state);
        return rgamma (state, 1.0 + a, b) * pow (u, 1.0 / a);
      }
    
    {
      double x, v, u;
      double d = a - 1.0 / 3.0;
      double c = (1.0 / 3.0) / sqrt (d);
      
      while (1){
        do{
          x = curand_normal_double(state);
          v = 1.0 + c * x;
        } while (v <= 0);
        
        v = v * v * v;
        u = curand_uniform_double(state);
        
        if (u < 1 - 0.0331 * x * x * x * x) 
          break;
        
        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
          break;
      }
      return b * d * v;
    }
  }

__device__ double rbeta(curandState *state, const double a, const double b){
  
  double x,y;
  x = rgamma(state, a, 1);
  y = rgamma(state, b, 1);
  
  return x/(x+y);
}


__global__ void setup_kernel(curandState *state) {
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence number, no offset */
    
    curand_init(1234, id, 0, &state[id]);
}

__global__ void getBeta(curandState *states, double *a, double *b, double *result){
  
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  
  result[id] = rbeta(&(states[id]), a[id], b[id]);
}

#endif