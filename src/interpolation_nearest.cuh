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
       const T inter_coord_x) {
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  return u[ix_out];
}

template <typename T>
__device__ T cuda_interpolate1d_nearest_backward(
    const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, 
    const int NX, const float LX, const float hX,
    const int boundary,
    const T inter_coord_x,
    const T forward_val,
    torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> grad_u,
    T &grad_phi_idx
    //    torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_phi 
    ) {
  
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  atomicAdd( &grad_u[ix_out], forward_val);

  // Gradients wrt. the coordinates
  //grad_phi_idx += forward_val;
  //grad_phi_idx += u[ix_out] * forward_val;
  grad_phi_idx = 0;
  //atomicAdd( &grad_phi[ix_out][0], forward_val);

  return u[ix_out];
}


//=========================================================
// nearest interpolation in 2D
//=========================================================


//=====================
// scalar fields
//=====================
template <typename T>
__device__ T cuda_interpolate2d_nearest(
      const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
      const int NY, const int NX,
      const float LY, const float LX,
      const float hY, const float hX,
      const int boundary,
      const T inter_coord_y, const T inter_coord_x) {
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(inter_coord_y / hY);
  const float dist_y_f = inter_coord_y / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - inter_coord_y / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  return u[iy_out][ix_out];
}

template <typename T>
__device__ T cuda_interpolate2d_nearest_backward(
    const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u, 
    const int NY, const int NX,
    const float LY, const float LX,
    const float hY, const float hX,
    const int boundary,
    const T inter_coord_y, const T inter_coord_x,
    const T forward_val,
    torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_u,
    T &grad_phi_idy, T &grad_phi_idx
    //    torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_phi 
    ) {
  
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(inter_coord_y / hY);
  const float dist_y_f = inter_coord_y / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - inter_coord_y / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  atomicAdd( &grad_u[iy_out][ix_out], forward_val);

  // Gradients wrt. the coordinates
  // grad_phi_idx += forward_val;
  // grad_phi_idy += forward_val;
  grad_phi_idx = 0;
  grad_phi_idy = 0;

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
                                         const T inter_coord_y, const T inter_coord_x,
                                         const int comp) {
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(inter_coord_y / hY);
  const float dist_y_f = inter_coord_y / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - inter_coord_y / hY;
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
                                         const T inter_coord_y, const T inter_coord_x,
                                         const int comp_i, const int comp_j ) {
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(inter_coord_y / hY);
  const float dist_y_f = inter_coord_y / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - inter_coord_y / hY;
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
__device__ T cuda_interpolate3d_nearest(
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
    const int NZ, const int NY, const int NX,
    const float LZ, const float LY, const float LX,
    const float hZ, const float hY, const float hX,
    const int boundary,
    const T inter_coord_z, const T inter_coord_y, const T inter_coord_x) {
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(inter_coord_y / hY);
  const float dist_y_f = inter_coord_y / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - inter_coord_y / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  const int iz_f = floorf(inter_coord_z / hZ);
  const float dist_z_f = inter_coord_z / hZ - iz_f;
  const int iz_c = iz_f + 1;
  const float dist_z_c = iz_c - inter_coord_z / hZ;
  int iz_out;
  if(dist_z_f < dist_z_c ){
    iz_out = getIndexInterpolate(iz_f,NZ,boundary);
  }else{
    iz_out = getIndexInterpolate(iz_c,NZ,boundary);
  }

  return u[iz_out][iy_out][ix_out];
}



template <typename T>
__device__ T cuda_interpolate3d_nearest_backward(
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u, 
    const int NZ, const int NY, const int NX,
    const float LZ, const float LY, const float LX,
    const float hZ, const float hY, const float hX,
    const int boundary,
    const T inter_coord_z, const T inter_coord_y, const T inter_coord_x,
    const T forward_val,
    torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_u,
    T &grad_phi_idz, T &grad_phi_idy, T &grad_phi_idx
    //    torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_phi 
    ) {
  
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(inter_coord_y / hY);
  const float dist_y_f = inter_coord_y / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - inter_coord_y / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  const int iz_f = floorf(inter_coord_z / hZ);
  const float dist_z_f = inter_coord_z / hZ - iz_f;
  const int iz_c = iz_f + 1;
  const float dist_z_c = iz_c - inter_coord_z / hZ;
  int iz_out;
  if(dist_z_f < dist_z_c ){
    iz_out = getIndexInterpolate(iz_f,NZ,boundary);
  }else{
    iz_out = getIndexInterpolate(iz_c,NZ,boundary);
  }

  // Gradients wrt. image
  atomicAdd( &grad_u[iz_out][iy_out][ix_out], forward_val);

  // Gradients wrt. the coordinates
  // grad_phi_idx += forward_val;
  // grad_phi_idy += forward_val;
  // grad_phi_idz += forward_val;
  grad_phi_idx = 0;
  grad_phi_idy = 0;
  grad_phi_idz = 0;

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
                                          const T inter_coord_z, const T inter_coord_y, const T inter_coord_x, 
                                          const int comp) {
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(inter_coord_y / hY);
  const float dist_y_f = inter_coord_y / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - inter_coord_y / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  const int iz_f = floorf(inter_coord_z / hZ);
  const float dist_z_f = inter_coord_z / hZ - iz_f;
  const int iz_c = iz_f + 1;
  const float dist_z_c = iz_c - inter_coord_z / hZ;
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
                                          const T inter_coord_z, const T inter_coord_y, const T inter_coord_x, 
                                          const int comp_i, const int comp_j ) {
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(inter_coord_y / hY);
  const float dist_y_f = inter_coord_y / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - inter_coord_y / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  const int iz_f = floorf(inter_coord_z / hZ);
  const float dist_z_f = inter_coord_z / hZ - iz_f;
  const int iz_c = iz_f + 1;
  const float dist_z_c = iz_c - inter_coord_z / hZ;
  int iz_out;
  if(dist_z_f < dist_z_c ){
    iz_out = getIndexInterpolate(iz_f,NZ,boundary);
  }else{
    iz_out = getIndexInterpolate(iz_c,NZ,boundary);
  }

  return u[iz_out][iy_out][ix_out][comp_i][comp_j];
}


//=====================
// CNN
//=====================
template <typename T>
__device__ T cuda_interpolateCNN3d_nearest(const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
                                          const int batch, const int channel,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const int boundary,
                                          const T inter_coord_z, const T inter_coord_y, const T inter_coord_x ) {
  const int ix_f = floorf(inter_coord_x / hX);
  const float dist_x_f = inter_coord_x / hX - ix_f;
  const int ix_c = ix_f + 1;
  const float dist_x_c = ix_c - inter_coord_x / hX;
  int ix_out;
  if(dist_x_f < dist_x_c ){
    ix_out = getIndexInterpolate(ix_f,NX,boundary);
  }else{
    ix_out = getIndexInterpolate(ix_c,NX,boundary);
  }

  const int iy_f = floorf(inter_coord_y / hY);
  const float dist_y_f = inter_coord_y / hY - iy_f;
  const int iy_c = iy_f + 1;
  const float dist_y_c = iy_c - inter_coord_y / hY;
  int iy_out;
  if(dist_y_f < dist_y_c ){
    iy_out = getIndexInterpolate(iy_f,NY,boundary);
  }else{
    iy_out = getIndexInterpolate(iy_c,NY,boundary);
  }

  const int iz_f = floorf(inter_coord_z / hZ);
  const float dist_z_f = inter_coord_z / hZ - iz_f;
  const int iz_c = iz_f + 1;
  const float dist_z_c = iz_c - inter_coord_z / hZ;
  int iz_out;
  if(dist_z_f < dist_z_c ){
    iz_out = getIndexInterpolate(iz_f,NZ,boundary);
  }else{
    iz_out = getIndexInterpolate(iz_c,NZ,boundary);
  }

  return u[batch][channel][iz_out][iy_out][ix_out];
}

#endif