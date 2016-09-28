#include "util/include.h"
#include "util/cuda_usage.h"

extern "C" SEXP RgetDeviceCount(){
  int count = 0;
  cudaGetDeviceCount(&count);
  
  SEXP result = PROTECT(allocVector(INTSXP, 1));
  INTEGER(result)[0] = count;
  UNPROTECT(1);
  
  return result;
}

extern "C" SEXP RsetDevice(SEXP device) {
  SEXP result = PROTECT(allocVector(INTSXP, 1));
  INTEGER(result)[0] = setDevice(INTEGER(device)[0]);
  UNPROTECT(1);
  return result;
}

extern "C" SEXP RgetDevice(){
  int device = 0;
  cudaGetDevice(&device);
  
  SEXP result = PROTECT(allocVector(INTSXP, 1));
  INTEGER(result)[0] = device;
  UNPROTECT(1);
  
  return result;
}
