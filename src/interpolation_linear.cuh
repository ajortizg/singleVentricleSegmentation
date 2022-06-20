#ifndef __INTERPOLATION_LINEAR_CUH_
#define __INTERPOLATION_LINEAR_CUH_

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
// linear interpolation in 1D
//=========================================================

template <typename T>
__device__ T cuda_interpolate1d_linear(
       const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, 
       const int NX, const float LX, const float hX,
       const int boundary,
       const T inter_coord_x) {
  const int ix_f = floorf(inter_coord_x / hX);
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);  

  T u_f = u[ix_f_out];
  T u_c = u[ix_c_out];

  T out = (1 - wx) * u_f;
  out += wx * u_c;

  return out;
}

template <typename T>
__device__ T cuda_interpolate1d_linear_backward(
    const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, 
    const int NX, const float LX, const float hX,
    const int boundary,
    const T inter_coord_x,
    const T forward_val,
    torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> grad_u,
    T &grad_phi_idx
    //    torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_phi 
    ) {
  
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  T u_f = u[ix_f_out];
  T u_c = u[ix_c_out];

  T out = (1 - wx) * u_f;
  out += wx * u_c;

  // Gradients wrt. the pixel values
  atomicAdd( &(grad_u[ix_f_out]), (1 - wx) * forward_val);
  atomicAdd( &(grad_u[ix_c_out]), wx * forward_val );

  // Gradients wrt. the coordinates
  grad_phi_idx += (u_c - u_f) / hX * forward_val;

  return out;
}


//=========================================================
// bi-linear interpolation in 2D
//=========================================================


//=====================
// scalar fields
//=====================
template <typename T>
__device__ T cuda_interpolate2d_bilinear(const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const int boundary,
                                         const T inter_coord_y, const T inter_coord_x) {
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(inter_coord_y / hY );
  const int iy_c = iy_f + 1;
  const T wy = inter_coord_y / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  T u_ff = u[iy_f_out][ix_f_out];
  T u_fc = u[iy_c_out][ix_f_out];
  T u_cf = u[iy_f_out][ix_c_out];
  T u_cc = u[iy_c_out][ix_c_out];

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}

template <typename T>
__device__ T cuda_interpolate2d_bilinear_backward(
    const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u, 
    const int NY, const int NX, 
    const float LY, const float LX,
    const float hY, const float hX,
    const int boundary,
    const T inter_coord_y, const T inter_coord_x,
    const T forward_val,
    torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_u,
    T &grad_phi_idy, T &grad_phi_idx
    //    torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_phi 
    ) {
  
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(inter_coord_y / hY );
  const int iy_c = iy_f + 1;
  const T wy = inter_coord_y / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  T u_ff = u[iy_f_out][ix_f_out];
  T u_fc = u[iy_c_out][ix_f_out];
  T u_cf = u[iy_f_out][ix_c_out];
  T u_cc = u[iy_c_out][ix_c_out];

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  // Gradients wrt. the pixel values
  atomicAdd( &(grad_u[iy_f_out][ix_f_out]), (1 - wy) * (1 - wx) * forward_val );
  atomicAdd( &(grad_u[iy_c_out][ix_f_out]),  wy * (1 - wx) * forward_val );
  atomicAdd( &(grad_u[iy_f_out][ix_c_out]), (1 - wy) * wx * forward_val );
  atomicAdd( &(grad_u[iy_c_out][ix_c_out]), wy * wx * forward_val );

  // Gradients wrt. the coordinates
  grad_phi_idx += ((1 - wy) * (u_cf - u_ff) + wy * (u_cc - u_fc)) / hX * forward_val;
  grad_phi_idy += ((1 - wx) * (u_fc - u_ff) + wx * (u_cc - u_cf)) / hY * forward_val;

  return out;
}

//=====================
// vector fields
//=====================
template <typename T>
__device__ T cuda_interpolateVectorField2d_bilinear (const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const int boundary,
                                         const T inter_coord_y, const T inter_coord_x,
                                         const int comp) {
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(inter_coord_y / hY );
  const int iy_c = iy_f + 1;
  const T wy = inter_coord_y / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  T u_ff = u[iy_f_out][ix_f_out][comp];
  T u_fc = u[iy_c_out][ix_f_out][comp];
  T u_cf = u[iy_f_out][ix_c_out][comp];
  T u_cc = u[iy_c_out][ix_c_out][comp];

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}



//=====================
// matrix fields
//=====================
template <typename T>
__device__ T cuda_interpolateMatrixField2d_bilinear(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const int boundary,
                                         const T inter_coord_y, const T inter_coord_x,
                                         const int comp_i, const int comp_j ) {
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(inter_coord_y / hY );
  const int iy_c = iy_f + 1;
  const T wy = inter_coord_y / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  T u_ff = u[iy_f_out][ix_f_out][comp_i][comp_j];
  T u_fc = u[iy_c_out][ix_f_out][comp_i][comp_j];
  T u_cf = u[iy_f_out][ix_c_out][comp_i][comp_j];
  T u_cc = u[iy_c_out][ix_c_out][comp_i][comp_j];

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}


//=========================================================
// tri-linear interpolation in 3D
//=========================================================

//=====================
// scalar fields
//=====================
template <typename T>
__device__ T cuda_interpolate3d_trilinear(
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
    const int NZ, const int NY, const int NX,
    const float LZ, const float LY, const float LX,
    const float hZ, const float hY, const float hX,
    const int boundary,
    const T inter_coord_z, const T inter_coord_y, const T inter_coord_x) {
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(inter_coord_y / hY );
  const int iy_c = iy_f + 1;
  const T wy = inter_coord_y / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  const int iz_f = floorf(inter_coord_z / hZ );
  const int iz_c = iz_f + 1;
  const T wz = inter_coord_z / hZ - iz_f;
  const int iz_f_out = getIndexInterpolate(iz_f,NZ,boundary);
  const int iz_c_out = getIndexInterpolate(iz_c,NZ,boundary);

  T u_fff = u[iz_f_out][iy_f_out][ix_f_out];
  T u_ffc = u[iz_c_out][iy_f_out][ix_f_out];
  T u_fcf = u[iz_f_out][iy_c_out][ix_f_out];
  T u_fcc = u[iz_c_out][iy_c_out][ix_f_out];
  T u_cff = u[iz_f_out][iy_f_out][ix_c_out];
  T u_cfc = u[iz_c_out][iy_f_out][ix_c_out];
  T u_ccf = u[iz_f_out][iy_c_out][ix_c_out];
  T u_ccc = u[iz_c_out][iy_c_out][ix_c_out];

  T out = (1 - wz) * (1 - wy) * (1 - wx) * u_fff;
  out += wz * (1 - wy) * (1 - wx) * u_ffc;
  out += (1 - wz) * (1 - wy) * wx * u_cff;
  out += wz * (1 - wy) * wx * u_cfc;
  out += (1 - wz) * wy * (1 - wx) * u_fcf;
  out += wz * wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wz * wy * wx * u_ccc;

  return out;
}


template <typename T>
__device__ T cuda_interpolate3d_trilinear_backward(
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u, 
    const int NZ, const int NY, const int NX,
    const float LZ, const float LY, const float LX,
    const float hZ, const float hY, const float hX,
    const int boundary,
    const T inter_coord_z, const T inter_coord_y, const T inter_coord_x,
    const T forward_val,
    torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_u,
    T &grad_phi_idz, T &grad_phi_idy, T &grad_phi_idx
    //    torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_phi 
    ) {
  
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(inter_coord_y / hY );
  const int iy_c = iy_f + 1;
  const T wy = inter_coord_y / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  const int iz_f = floorf(inter_coord_z / hZ );
  const int iz_c = iz_f + 1;
  const T wz = inter_coord_z / hZ - iz_f;
  const int iz_f_out = getIndexInterpolate(iz_f,NZ,boundary);
  const int iz_c_out = getIndexInterpolate(iz_c,NZ,boundary);

  T u_fff = u[iz_f_out][iy_f_out][ix_f_out];
  T u_ffc = u[iz_c_out][iy_f_out][ix_f_out];
  T u_fcf = u[iz_f_out][iy_c_out][ix_f_out];
  T u_fcc = u[iz_c_out][iy_c_out][ix_f_out];
  T u_cff = u[iz_f_out][iy_f_out][ix_c_out];
  T u_cfc = u[iz_c_out][iy_f_out][ix_c_out];
  T u_ccf = u[iz_f_out][iy_c_out][ix_c_out];
  T u_ccc = u[iz_c_out][iy_c_out][ix_c_out];

  T out = (1 - wz) * (1 - wy) * (1 - wx) * u_fff;
  out += wz * (1 - wy) * (1 - wx) * u_ffc;
  out += (1 - wz) * (1 - wy) * wx * u_cff;
  out += wz * (1 - wy) * wx * u_cfc;
  out += (1 - wz) * wy * (1 - wx) * u_fcf;
  out += wz * wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wz * wy * wx * u_ccc;

  // Gradients wrt. the pixel values
  atomicAdd( &(grad_u[iz_f_out][iy_f_out][ix_f_out]), (1 - wz) * (1 - wy) * (1 - wx) * forward_val );
  atomicAdd( &(grad_u[iz_f_out][iy_c_out][ix_f_out]), (1 - wz) *  wy * (1 - wx) * forward_val );
  atomicAdd( &(grad_u[iz_f_out][iy_f_out][ix_c_out]), (1 - wz) * (1 - wy) * wx * forward_val );
  atomicAdd( &(grad_u[iz_f_out][iy_c_out][ix_c_out]), (1 - wz) * wy * wx * forward_val );
  atomicAdd( &(grad_u[iz_c_out][iy_f_out][ix_f_out]), wz * (1 - wy) * (1 - wx) * forward_val );
  atomicAdd( &(grad_u[iz_c_out][iy_c_out][ix_f_out]), wz * wy * (1 - wx) * forward_val );
  atomicAdd( &(grad_u[iz_c_out][iy_f_out][ix_c_out]), wz * (1 - wy) * wx * forward_val );
  atomicAdd( &(grad_u[iz_c_out][iy_c_out][ix_c_out]), wz * wy * wx * forward_val );

  // Gradients wrt. the coordinates
  grad_phi_idx += ((1 - wz) * (1 - wy) * (u_cff - u_fff) + (1 - wz) * wy * (u_ccf - u_fcf) + wz * (1 - wy) * (u_cfc - u_ffc) + wz * wy * (u_ccc - u_fcc) ) / hX * forward_val;
  grad_phi_idy += ((1 - wz) * (1 - wx) * (u_fcf - u_fff) + (1 - wz) * wx * (u_ccf - u_cff) + wz * (1 - wx) * (u_fcc - u_ffc) + wz * wx * (u_ccc - u_cfc) ) / hY * forward_val;
  grad_phi_idz += ((1 - wy) * (1 - wx) * (u_ffc - u_fff) + (1 - wy) * wx * (u_cfc - u_cff) + wy * (1 - wx) * (u_fcc - u_fcf) + wy * wx * (u_ccc - u_ccf) ) / hZ * forward_val;

  return out;
}

//=====================
// vector fields
//=====================
template <typename T>
__device__ T cuda_interpolateVectorField3d_trilinear(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const int boundary,
                                          const T inter_coord_z, const T inter_coord_y, const T inter_coord_x, 
                                          const int comp) {
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(inter_coord_y / hY );
  const int iy_c = iy_f + 1;
  const T wy = inter_coord_y / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  const int iz_f = floorf(inter_coord_z / hZ );
  const int iz_c = iz_f + 1;
  const T wz = inter_coord_z / hZ - iz_f;
  const int iz_f_out = getIndexInterpolate(iz_f,NZ,boundary);
  const int iz_c_out = getIndexInterpolate(iz_c,NZ,boundary);

  T u_fff = u[iz_f_out][iy_f_out][ix_f_out][comp];
  T u_ffc = u[iz_c_out][iy_f_out][ix_f_out][comp];
  T u_fcf = u[iz_f_out][iy_c_out][ix_f_out][comp];
  T u_fcc = u[iz_c_out][iy_c_out][ix_f_out][comp];
  T u_cff = u[iz_f_out][iy_f_out][ix_c_out][comp];
  T u_cfc = u[iz_c_out][iy_f_out][ix_c_out][comp];
  T u_ccf = u[iz_f_out][iy_c_out][ix_c_out][comp];
  T u_ccc = u[iz_c_out][iy_c_out][ix_c_out][comp];

  T out = (1 - wz) * (1 - wy) * (1 - wx) * u_fff;
  out += wz * (1 - wy) * (1 - wx) * u_ffc;
  out += (1 - wz) * (1 - wy) * wx * u_cff;
  out += wz * (1 - wy) * wx * u_cfc;
  out += (1 - wz) * wy * (1 - wx) * u_fcf;
  out += wz * wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wz * wy * wx * u_ccc;

  return out;
}


//=====================
// matrix fields
//=====================
template <typename T>
__device__ T cuda_interpolateMatrixField3d_trilinear(const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const int boundary,
                                          const T inter_coord_z, const T inter_coord_y, const T inter_coord_x, 
                                          const int comp_i, const int comp_j ) {
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(inter_coord_y / hY );
  const int iy_c = iy_f + 1;
  const T wy = inter_coord_y / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  const int iz_f = floorf(inter_coord_z / hZ );
  const int iz_c = iz_f + 1;
  const T wz = inter_coord_z / hZ - iz_f;
  const int iz_f_out = getIndexInterpolate(iz_f,NZ,boundary);
  const int iz_c_out = getIndexInterpolate(iz_c,NZ,boundary);

  T u_fff = u[iz_f_out][iy_f_out][ix_f_out][comp_i][comp_j];
  T u_ffc = u[iz_c_out][iy_f_out][ix_f_out][comp_i][comp_j];
  T u_fcf = u[iz_f_out][iy_c_out][ix_f_out][comp_i][comp_j];
  T u_fcc = u[iz_c_out][iy_c_out][ix_f_out][comp_i][comp_j];
  T u_cff = u[iz_f_out][iy_f_out][ix_c_out][comp_i][comp_j];
  T u_cfc = u[iz_c_out][iy_f_out][ix_c_out][comp_i][comp_j];
  T u_ccf = u[iz_f_out][iy_c_out][ix_c_out][comp_i][comp_j];
  T u_ccc = u[iz_c_out][iy_c_out][ix_c_out][comp_i][comp_j];

  T out = (1 - wz) * (1 - wy) * (1 - wx) * u_fff;
  out += wz * (1 - wy) * (1 - wx) * u_ffc;
  out += (1 - wz) * (1 - wy) * wx * u_cff;
  out += wz * (1 - wy) * wx * u_cfc;
  out += (1 - wz) * wy * (1 - wx) * u_fcf;
  out += wz * wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wz * wy * wx * u_ccc;

  return out;
}

//=====================
// CNN
//=====================
template <typename T>
__device__ T cuda_interpolateCNN3d_trilinear(const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
                                          const int batch, const int channel,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const int boundary,
                                          const T inter_coord_z, const T inter_coord_y, const T inter_coord_x ) {
  const int ix_f = floorf(inter_coord_x / hX );
  const int ix_c = ix_f + 1;
  const T wx = inter_coord_x / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(inter_coord_y / hY );
  const int iy_c = iy_f + 1;
  const T wy = inter_coord_y / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  const int iz_f = floorf(inter_coord_z / hZ );
  const int iz_c = iz_f + 1;
  const T wz = inter_coord_z / hZ - iz_f;
  const int iz_f_out = getIndexInterpolate(iz_f,NZ,boundary);
  const int iz_c_out = getIndexInterpolate(iz_c,NZ,boundary);

  T u_fff = u[batch][channel][iz_f_out][iy_f_out][ix_f_out];
  T u_ffc = u[batch][channel][iz_c_out][iy_f_out][ix_f_out];
  T u_fcf = u[batch][channel][iz_f_out][iy_c_out][ix_f_out];
  T u_fcc = u[batch][channel][iz_c_out][iy_c_out][ix_f_out];
  T u_cff = u[batch][channel][iz_f_out][iy_f_out][ix_c_out];
  T u_cfc = u[batch][channel][iz_c_out][iy_f_out][ix_c_out];
  T u_ccf = u[batch][channel][iz_f_out][iy_c_out][ix_c_out];
  T u_ccc = u[batch][channel][iz_c_out][iy_c_out][ix_c_out];

  T out = (1 - wz) * (1 - wy) * (1 - wx) * u_fff;
  out += wz * (1 - wy) * (1 - wx) * u_ffc;
  out += (1 - wz) * (1 - wy) * wx * u_cff;
  out += wz * (1 - wy) * wx * u_cfc;
  out += (1 - wz) * wy * (1 - wx) * u_fcf;
  out += wz * wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wz * wy * wx * u_ccc;

  return out;
}

#endif