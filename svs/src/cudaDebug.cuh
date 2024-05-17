#ifndef __CUDADEBUG_CUH_
#define __CUDADEBUG_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 
#include <iostream>
#include <stdio.h>

// for debugging
// #define CUDA_ERROR_CHECK
// #define CUDA_TIMING

#define cudaSafeCall( err ) __cnnCudaSafeCall( err, __FILE__, __LINE__ )

inline void __cnnCudaSafeCall( cudaError_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
#endif
  return;
}


#ifdef CUDA_TIMING
class CudaTimer
{
public:
  CudaTimer() 
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~CudaTimer() 
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() 
  {
    cudaEventRecord(start_, 0);
  }

  float elapsed() 
  {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float t = 0;
    cudaEventElapsedTime(&t, start_, stop_);
    return t;
  }

private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};
#endif


#endif
