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
       const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX);
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);  

  T u_f = u[ix_f_out];
  T u_c = u[ix_c_out];

  T out = (1 - wx) * u_f;
  out += wx * u_c;

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
                                         const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;
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

//=====================
// vector fields
//=====================
template <typename T>
__device__ T cuda_interpolateVectorField2d_bilinear (const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const int boundary,
                                         const T coord_y_warped, const T coord_x_warped,
                                         const int comp) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;
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
                                         const T coord_y_warped, const T coord_x_warped,
                                         const int comp_i, const int comp_j ) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;
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
__device__ T cuda_interpolate3d_trilinear(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const int boundary,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;
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


//=====================
// vector fields
//=====================
template <typename T>
__device__ T cuda_interpolateVectorField3d_trilinear(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const int boundary,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped, 
                                          const int comp) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;
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
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped, 
                                          const int comp_i, const int comp_j ) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;
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

#endif