#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/reduce.h>

double my_reduce(thrust::host_vector<double> hvec){
  thrust::device_vector<double> dvec(hvec);
  double out = reduce(dvec.begin(), dvec.end());
  return out;
}