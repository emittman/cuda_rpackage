
/*
#include "iterator2.h"
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <iostream>

int main(){

  thrust::device_vector<int> vec(10);
  thrust::device_vector<int> vec2(2);
  thrust::device_vector<int> out(20);
  
  //thrust::sequence(vec.begin(), vec.end(), 0, 1);
  //thrust::sequence(vec2.begin(), vec2.end(), 0, 1);
  
  std::cout << "initialized vec:\n";
  
  //thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, " "));
  
  /*
  std::cout << "initialized vec2:\n";
  thrust::copy(vec2.begin(), vec2.end(), std::ostream_iterator<int>(std::cout, " "));
  
  std::cout <<"\n repEach(5, 2), vec";
  gRepEach<intIter>::iterator eachit = getGRepEachIter(vec.begin(), vec.end(), 5, 2);
  thrust::copy(eachit, eachit + 20, out.begin());
  thrust::copy(out.begin(), out.end(), std::ostream_iterator<int>(std::cout, " "));
  
  std::cout <<"\n repEach(5, 2), vec2";
  eachit = getGRepEachIter(vec2.begin(), vec2.end(), 5, 2);
  thrust::copy(eachit, eachit + 20, out.begin());
  thrust::copy(out.begin(), out.end(), std::ostream_iterator<int>(std::cout, " "));
  */

  return 0;
}*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

int main(void)
{
    // H has storage for 4 integers
    thrust::host_vector<int> H(4);

    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;
    
    // H.size() returns the size of vector H
    std::cout << "H has size " << H.size() << std::endl;

    // print contents of H
    for(int i = 0; i < H.size(); i++)
        std::cout << "H[" << i << "] = " << H[i] << std::endl;

    // resize H
    H.resize(2);
    
    std::cout << "H now has size " << H.size() << std::endl;

    // Copy host_vector H to device_vector D
    thrust::device_vector<int> D = H;
    
    // elements of D can be modified
    D[0] = 99;
    D[1] = 88;
    
    // print contents of D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    // H and D are automatically deleted when the function returns
    return 0;
}
