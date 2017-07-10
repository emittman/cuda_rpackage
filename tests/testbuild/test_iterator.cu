#include "iterator2.h"
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <iostream>

int main(){

  ivec vec(10);
  ivec vec2(2);
  ivec out(20);
  thrust::sequence(vec.begin(), vec.end(), 0, 1);
  thrust::sequence(vec2.begin(), vec2.end(), 0, 1);
  std::cout << "initialized vec:\n";
  thrust::copy(vec.begin(), vec.out(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << "initialized vec2:\n";
  thrust::copy(vec2.begin(), vec2.out(), std::ostream_iterator<int>(std::cout, " "));
  
  std::cout <<"\n repEach(5, 2), vec"
  gRepEach<ivec>::iterator eachit = getRepEachIter(vec.begin(), vec.end(), 5, 2);
  thrust::copy(eachit, eachit + 20, out.begin());
  thrust::copy(out.begin(), out.out(), std::ostream_iterator<int>(std::cout, " "));
  
  std::cout <<"\n repEach(5, 2), vec2"
  gRepEach<ivec>::iterator eachit = getRepEachIter(vec2.begin(), vec2.end(), 5, 2);
  thrust::copy(eachit, eachit + 20, out.begin());
  thrust::copy(out.begin(), out.out(), std::ostream_iterator<int>(std::cout, " "));

  return 0;
}