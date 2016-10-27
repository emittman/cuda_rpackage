#ifndef PRINT_H
#define PRINT_H

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<iostream>

/*courtesy of https://github.com/thrust/thrust/blob/master/examples/expand.cu */

template <typename Vector>
void printVec(const Vector& v, int d1, int d2)
{
  typedef typename Vector::value_type T;
  for(int i=0; i<d2; ++i){
    thrust::copy(v.begin() + i*d1, v.begin() + (i+1)*d1, std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
  }
}

#endif