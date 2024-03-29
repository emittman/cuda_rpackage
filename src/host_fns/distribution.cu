#include "../header/distribution.h"

__device__ double rgamma(curandState *state, double a, double b, bool logscale = false){
  //case a >= 1
  double d = a - 1.0 / 3;
  double Y, U, v;
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
      if(logscale) return log(d) + log(v) - log(b);
      else return d * v / b;
    }
  }
}

__device__ double rgamma2(curandState *state, double a, double b, bool logscale = false){
  //case a < 1
  double u, x;
  u = 1/a * log(curand_uniform(state));
  x = rgamma(state, a + 1, b, true);
  if(logscale) return u + x;
  else return exp(u + x);
}

__device__ double rbeta(curandState *state,  double a, double b, bool logscale = false){
  
  double x,y,m,out;
  if(a<1){
    x = rgamma2(state, a, 1.0, true);
  } else{
    x = rgamma(state, a, 1.0, true);
  }
  if(b<1){
    y = rgamma2(state, b, 1.0, true);
  } else{
    y = rgamma(state, b, 1.0, true);
  }
  m = max(x,y);
  if(logscale){
    out = x - log(exp(x-m)+exp(y-m)) - m;
  }
  if(!logscale){
    out = exp(x - log(exp(x-m)+exp(y-m)) - m);
  }
  return out;
}

__global__ void setup_kernel(int seed, int n_threads, curandState *states) {
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < n_threads){
  /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(seed, id, 0, &states[id]);
  }
}

__global__ void getGamma(curandState *states, int n_threads, double *a, double *b, double *result, bool logscale = false){
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < n_threads){
    if(a[id]>=1){
      result[id] = rgamma(&(states[id]), a[id], b[id], logscale);
    } else {
      result[id] = rgamma2(&(states[id]), a[id], b[id], logscale);
    }
  }
}


__global__ void getBeta(curandState *states, int n_threads, double *a, double *b, double *result, bool logscale=false){
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < n_threads){
    result[id] = rbeta(&(states[id]), a[id], b[id], logscale);
  }
}

__global__ void getUniform(curandState *states, int n_threads, double *upper_result){

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < n_threads){
    upper_result[id] = log(curand_uniform(&(states[id]))) + upper_result[id];
  }
}

__global__ void getExponential(curandState *states, int n_threads, double *weights, double *result){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < n_threads){
    result[id] = -log(curand_uniform(&(states[id]))) / weights[id];
  }
}

__global__ void getNormal(curandState *states, int n_threads, double *result)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < n_threads){
    result[id] = curand_normal(&(states[id]));
  }
}
