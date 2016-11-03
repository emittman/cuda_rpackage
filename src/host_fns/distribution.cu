#include "../header/distribution.h"

__device__ double rgamma(curandState *state, double a, double b){

  float d = a - 1.0 / 3;
  float Y, U, v;
  while(true){
    // Generate a standard normal random variable
    Y = curand_normal(state);
    
    v = pow((1 + Y / sqrt(9 * d)), 3);
    
    // Necessary to avoid taking the log of a negative number later
    if(v <= 0) 
      continue;
    
    // Generate a standard uniform random variable
    U = curand_uniform(state);
    
    // Accept proposed Gamma random variable under following condition,
    // otherise repeat the loop
    if(log(U) < 0.5 * pow(Y,2) + d * (1 - v + log(v)) ){
      return d * v / b;
    }
  }
}

__device__ double rbeta(curandState *state,  double a, double b){
  
  double x,y;

  x = rgamma(state, a, 1.0);
  y = rgamma(state, b, 1.0);
  
  return x/(x+y);
}

__global__ void setup_kernel(curandState *states) {
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence number, no offset */
    
    curand_init(1234, id, 0, &states[id]);
}

__global__ void getBeta(curandState *states, double *a, double *b, double *result){
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  result[id] = rbeta(&(states[id]), a[id], b[id]);
}