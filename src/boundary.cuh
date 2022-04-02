#ifndef __BOUNDARY_CUH_
#define __BOUNDARY_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 
#include <iostream>
#include <stdio.h>


/** Boundary types. */
typedef enum
{
  //NEAREST: a a | a b c d e | e e
  BOUNDARY_NEAREST = 0,
  //MIRROR: b a | a b c d e | e d
  BOUNDARY_MIRROR = 1,
  //REFLECT: c b | a b c d e | d c
  BOUNDARY_REFLECT = 2
} BoundaryType;


//NEAREST Boundary
inline __device__ int getIndex_interpolate_bdryNearest( int ix_in, int N) 
{
  if( ix_in < 0 ) return 0;
  else if( ix_in > N - 1 ) return N-1;
  else return ix_in;
}

//MIRROR Boundary
inline __device__ int getIndex_interpolate_bdryMirror( int ix_in, int N) 
{
  if( ix_in < 0 ) return -ix_in - 1;
  else if( ix_in > N - 1 ) return 2*N-1-ix_in;
  else return ix_in;
}

//REFLECTION Boundary
inline __device__ int getIndex_interpolate_bdryReflect( int ix_in, int N) 
{
  if( ix_in < 0 ) return -ix_in;
  else if( ix_in > N - 1 ) return 2*N-2-ix_in;
  else return ix_in;
}





typedef int (*getIndex_interpolate_func) (int, int);

__device__ getIndex_interpolate_func func[3] = { getIndex_interpolate_bdryNearest, getIndex_interpolate_bdryMirror, getIndex_interpolate_bdryReflect };

inline __device__ int getIndexInterpolate ( int ix_in, int N, int op)
{
  return func[op](ix_in, N);
}


#endif
