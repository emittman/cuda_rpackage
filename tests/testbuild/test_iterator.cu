#include "iterator2.h"
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>

int main(){

  thrust::device_vector<int> vec(10);
  thrust::device_vector<int> vec2(2);
  thrust::device_vector<int> out(20);
  
  thrust::sequence(vec.begin(), vec.end(), 0, 1);
  thrust::transform(vec.begin(), vec.end(), vec.begin(), thrust::placeholders::_1 + 0.1);
  thrust::sequence(vec2.begin(), vec2.end(), 0, 1);
  thrust::transform(vec2.begin(), vec2.end(), vec2.begin(), thrust::placeholders::_1 + 0.1);

  std::cout << "initialized vec:\n";
  
  thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, " "));
  
  std::cout << "initialized vec2:\n";
  thrust::copy(vec2.begin(), vec2.end(), std::ostream_iterator<int>(std::cout, " "));
  
  std::cout <<"\n repEach(5, 2), vec\n";
  gRepEach<intIter>::iterator eachit = getGRepEachIter(vec.begin(), vec.end(), 5, 2);
  thrust::copy(eachit, eachit + 20, out.begin());
  thrust::copy(out.begin(), out.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout <<"\n repEach(5, 2), vec2\n";
  eachit = getGRepEachIter(vec2.begin(), vec2.end(), 5, 2);
  thrust::copy(eachit, eachit + 20, out.begin());
  thrust::copy(out.begin(), out.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\n";

  return 0;
}

