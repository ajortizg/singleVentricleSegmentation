#ifndef __INTERPOLATION_NEAREST_CUH_
#define __INTERPOLATION_NEAREST_CUH_

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 

#include <iostream>
#include <stdio.h>

#include "coreDefines.h"
#include "boundary.cuh"



//=========================================================
// nearest interpolation in 1D
//=========================================================

template <typename T>
__device__ T cuda_interpolate1d_nearest(
       const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, 
       const int NX, const float LX, const float hX,
       const int boundary,
       const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX);
  const float dist_x_f = coord_x_warped / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - coord_x_warped / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  return u[ix_out];
}


//=========================================================
// nearest interpolation in 2D
//=========================================================


//=====================
// scalar fields
//=====================
template <typename T>
__device__ T cuda_interpolate2d_nearest(const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const int boundary,
                                         const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX);
  const float dist_x_f = coord_x_warped / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - coord_x_warped / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(coord_y_warped / hY);
  const float dist_y_f = coord_y_warped / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - coord_y_warped / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  return u[iy_out][ix_out];
}

//=====================
// vector fields
//=====================
template <typename T>
__device__ T cuda_interpolateVectorField2d_nearest (const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const int boundary,
                                         const T coord_y_warped, const T coord_x_warped,
                                         const int comp) {
  const int ix_f = floorf(coord_x_warped / hX);
  const float dist_x_f = coord_x_warped / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - coord_x_warped / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(coord_y_warped / hY);
  const float dist_y_f = coord_y_warped / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - coord_y_warped / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }


  return u[iy_out][ix_out][comp];
}



//=====================
// matrix fields
//=====================
template <typename T>
__device__ T cuda_interpolateMatrixField2d_nearest(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const int boundary,
                                         const T coord_y_warped, const T coord_x_warped,
                                         const int comp_i, const int comp_j ) {
  const int ix_f = floorf(coord_x_warped / hX);
  const float dist_x_f = coord_x_warped / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - coord_x_warped / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(coord_y_warped / hY);
  const float dist_y_f = coord_y_warped / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - coord_y_warped / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }


  return u[iy_out][ix_out][comp_i][comp_j];
}


//=========================================================
// nearest interpolation in 3D
//=========================================================

//=====================
// scalar fields
//=====================
template <typename T>
__device__ T cuda_interpolate3d_nearest(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const int boundary,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX);
  const float dist_x_f = coord_x_warped / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - coord_x_warped / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(coord_y_warped / hY);
  const float dist_y_f = coord_y_warped / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - coord_y_warped / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  const int iz_f = floorf(coord_z_warped / hZ);
  const float dist_z_f = coord_z_warped / hZ - iz_f;
  const int iz_c = iz_f + 1;
  const float dist_z_c = iz_c - coord_z_warped / hZ;
  int iz_out;
  if(dist_z_f < dist_z_c ){
    iz_out = getIndexInterpolate(iz_f,NZ,boundary);
  }else{
    iz_out = getIndexInterpolate(iz_c,NZ,boundary);
  }


  return u[iz_out][iy_out][ix_out];
}


//=====================
// vector fields
//=====================
template <typename T>
__device__ T cuda_interpolateVectorField3d_nearest(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const int boundary,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped, 
                                          const int comp) {
  const int ix_f = floorf(coord_x_warped / hX);
  const float dist_x_f = coord_x_warped / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - coord_x_warped / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(coord_y_warped / hY);
  const float dist_y_f = coord_y_warped / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - coord_y_warped / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  const int iz_f = floorf(coord_z_warped / hZ);
  const float dist_z_f = coord_z_warped / hZ - iz_f;
  const int iz_c = iz_f + 1;
  const float dist_z_c = iz_c - coord_z_warped / hZ;
  int iz_out;
  if(dist_z_f < dist_z_c ){
    iz_out = getIndexInterpolate(iz_f,NZ,boundary);
  }else{
    iz_out = getIndexInterpolate(iz_c,NZ,boundary);
  }


  return u[iz_out][iy_out][ix_out][comp];
}


//=====================
// matrix fields
//=====================
template <typename T>
__device__ T cuda_interpolateMatrixField3d_nearest(const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const int boundary,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped, 
                                          const int comp_i, const int comp_j ) {
  const int ix_f = floorf(coord_x_warped / hX);
  const float dist_x_f = coord_x_warped / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - coord_x_warped / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(coord_y_warped / hY);
  const float dist_y_f = coord_y_warped / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - coord_y_warped / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  const int iz_f = floorf(coord_z_warped / hZ);
  const float dist_z_f = coord_z_warped / hZ - iz_f;
  const int iz_c = iz_f + 1;
  const float dist_z_c = iz_c - coord_z_warped / hZ;
  int iz_out;
  if(dist_z_f < dist_z_c ){
    iz_out = getIndexInterpolate(iz_f,NZ,boundary);
  }else{
    iz_out = getIndexInterpolate(iz_c,NZ,boundary);
  }


  return u[iz_out][iy_out][ix_out][comp_i][comp_j];
}

#endif