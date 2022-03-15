#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 

#include <iostream>
#include <stdio.h>

#include "coreDefines.h"


//=============================
// (multi-)linear interpolation
template <typename T>
__device__ T cuda_interpolate1d_linear(
       const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, 
       const int NX, const float LX, const float hX,
       const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX);
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  T u_f = 0;
  if (ix_f >= 0 && ix_f < NX) {
      u_f = u[ix_f];
  }

  T u_c = 0;
  if (ix_c >= 0 && ix_c < NX) {
      u_c = u[ix_c];
  }

  T out = (1 - wx) * u_f;
  out += wx * u_c;

  return out;
}


template <typename T>
__device__ T cuda_interpolate2d_bilinear(const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  T u_ff = 0, u_fc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_ff = u[iy_f][ix_f];

    if (iy_c >= 0 && iy_c < NY)
      u_fc = u[iy_c][ix_f];
  }

  T u_cf = 0, u_cc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_cf = u[iy_f][ix_c];

    if (iy_c >= 0 && iy_c < NY)
      u_cc = u[iy_c][ix_c];
  }

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}


template <typename T>
__device__ T cuda_interpolateVectorField2d_bilinear(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const T coord_y_warped, const T coord_x_warped,
                                         const int comp) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  T u_ff = 0, u_fc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_ff = u[iy_f][ix_f][comp];

    if (iy_c >= 0 && iy_c < NY)
      u_fc = u[iy_c][ix_f][comp];
  }

  T u_cf = 0, u_cc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_cf = u[iy_f][ix_c][comp];

    if (iy_c >= 0 && iy_c < NY)
      u_cc = u[iy_c][ix_c][comp];
  }

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}


template <typename T>
__device__ T cuda_interpolateMatrixField2d_bilinear(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const T coord_y_warped, const T coord_x_warped,
                                         const int comp_i, const int comp_j ) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  T u_ff = 0, u_fc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_ff = u[iy_f][ix_f][comp_i][comp_j];

    if (iy_c >= 0 && iy_c < NY)
      u_fc = u[iy_c][ix_f][comp_i][comp_j];
  }

  T u_cf = 0, u_cc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_cf = u[iy_f][ix_c][comp_i][comp_j];

    if (iy_c >= 0 && iy_c < NY)
      u_cc = u[iy_c][ix_c][comp_i][comp_j];
  }

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}


template <typename T>
__device__ T cuda_interpolate3d_trilinear(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;

  T u_fff = 0, u_ffc = 0, u_fcf = 0, u_fcc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fff = u[iz_f][iy_f][ix_f];

        if (iz_c >= 0 && iz_c < NZ)
          u_ffc = u[iz_c][iy_f][ix_f];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fcf = u[iz_f][iy_c][ix_f];

        if (iz_c >= 0 && iz_c < NZ)
          u_fcc = u[iz_c][iy_c][ix_f];
    }
  }

  T u_cff = 0, u_cfc = 0, u_ccf = 0, u_ccc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_cff = u[iz_f][iy_f][ix_c];

        if (iz_c >= 0 && iz_c < NZ)
          u_cfc = u[iz_c][iy_f][ix_c];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_ccf = u[iz_f][iy_c][ix_c];

        if (iz_c >= 0 && iz_c < NZ)
          u_ccc = u[iz_c][iy_c][ix_c];
    }
  }

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
__device__ T cuda_interpolateVectorField3d_trilinear(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped, 
                                          const int comp) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;

  T u_fff = 0, u_ffc = 0, u_fcf = 0, u_fcc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fff = u[iz_f][iy_f][ix_f][comp];

        if (iz_c >= 0 && iz_c < NZ)
          u_ffc = u[iz_c][iy_f][ix_f][comp];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fcf = u[iz_f][iy_c][ix_f][comp];

        if (iz_c >= 0 && iz_c < NZ)
          u_fcc = u[iz_c][iy_c][ix_f][comp];
    }
  }

  T u_cff = 0, u_cfc = 0, u_ccf = 0, u_ccc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_cff = u[iz_f][iy_f][ix_c][comp];

        if (iz_c >= 0 && iz_c < NZ)
          u_cfc = u[iz_c][iy_f][ix_c][comp];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_ccf = u[iz_f][iy_c][ix_c][comp];

        if (iz_c >= 0 && iz_c < NZ)
          u_ccc = u[iz_c][iy_c][ix_c][comp];
    }
  }

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
__device__ T cuda_interpolateMatrixField3d_trilinear(const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped, 
                                          const int comp_i, const int comp_j ) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;

  T u_fff = 0, u_ffc = 0, u_fcf = 0, u_fcc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fff = u[iz_f][iy_f][ix_f][comp_i][comp_j];

        if (iz_c >= 0 && iz_c < NZ)
          u_ffc = u[iz_c][iy_f][ix_f][comp_i][comp_j];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fcf = u[iz_f][iy_c][ix_f][comp_i][comp_j];

        if (iz_c >= 0 && iz_c < NZ)
          u_fcc = u[iz_c][iy_c][ix_f][comp_i][comp_j];
    }
  }

  T u_cff = 0, u_cfc = 0, u_ccf = 0, u_ccc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_cff = u[iz_f][iy_f][ix_c][comp_i][comp_j];

        if (iz_c >= 0 && iz_c < NZ)
          u_cfc = u[iz_c][iy_f][ix_c][comp_i][comp_j];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_ccf = u[iz_f][iy_c][ix_c][comp_i][comp_j];

        if (iz_c >= 0 && iz_c < NZ)
          u_ccc = u[iz_c][iy_c][ix_c][comp_i][comp_j];
    }
  }

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





//=============================
// cubic hermite spline interpolation

template <typename T>
__device__ T cuda_interpolate1d_cubicHermiteSpline_local(volatile T* localBuffer, const T local_coord) {

  const int kernel_size=4;

  const int ix_f = floorf(local_coord);
  const int ix_c = ix_f + 1;
  const int ix_f_1 = ix_f - 1;
  const int ix_c_1 = ix_c + 1;

  // get the input values
  T i_f = 0;
  if (ix_f >= 0 && ix_f < kernel_size) i_f = localBuffer[ix_f];
  T i_f_1 = 0;
  if (ix_f_1 >= 0 && ix_f_1 < kernel_size) i_f_1 = localBuffer[ix_f_1];
  T i_c = 0;
  if (ix_c >= 0 && ix_c < kernel_size) i_c = localBuffer[ix_c];
  T i_c_1 = 0;
  if (ix_c_1 >= 0 && ix_c_1 < kernel_size) i_c_1 = localBuffer[ix_c_1];

  // determine the coefficients
  const T p_f = i_f;
  const T p_prime_f = (i_c - i_f_1) / 2;
  const T p_c = i_c;
  const T p_prime_c = (i_c_1 - i_f) / 2;

  const T a = 2 * p_f - 2 * p_c + p_prime_f + p_prime_c;
  const T b = -3 * p_f + 3 * p_c - 2 * p_prime_f - p_prime_c;
  const T c = p_prime_f;
  const T d = p_f;

  const T wx = local_coord - ix_f;

  T out = wx * (wx * (wx * a + b) + c) + d;

  return out;
}


template <typename T>
__device__ T cuda_interpolate1d_cubicHermiteSpline(const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, 
                                      const int NX, const float LX, const float hX, const T coord_x_warped) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;
  T buff_x[4];

  for (int dx = -1; dx < 3; ++dx)
  {
        const int c_ix_x = ix_f + dx;
        if (c_ix_x >= 0 && c_ix_x < NX)
          buff_x[dx + 1] = u[c_ix_x];
        else
          buff_x[dx + 1] = 0;
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);

  return out;
}


template <typename T>
__device__ T cuda_interpolate2d_bicubicHermiteSpline(const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u, 
                                        const int NY, const int NX,
                                        const float LY, const float LX,
                                        const float hY, const float hX,
                                        const T coord_y_warped, const T coord_x_warped) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY);
  const T wy = coord_y_warped / hY - iy_f;

  T buff_y[4];
  T buff_x[4];

  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_ix_y = iy_f + dy;

    if (c_ix_y >= 0 && c_ix_y < NY)
    {
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_ix_x = ix_f + dx;
        if (c_ix_x >= 0 && c_ix_x < NX)
          buff_x[dx + 1] = u[c_ix_y][c_ix_x];
        else
          buff_x[dx + 1] = 0;
      }
      buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
    }
    else
      buff_y[dy + 1] = 0;
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);

  return out;
}


template <typename T>
__device__ T cuda_interpolateVectorField2d_bicubicHermiteSpline(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u, 
                                        const int NY, const int NX,
                                        const float LY, const float LX,
                                        const float hY, const float hX,
                                        const T coord_y_warped, const T coord_x_warped,
                                        const int comp) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY);
  const T wy = coord_y_warped / hY - iy_f;

  T buff_y[4];
  T buff_x[4];

  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_ix_y = iy_f + dy;

    if (c_ix_y >= 0 && c_ix_y < NY)
    {
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_ix_x = ix_f + dx;
        if (c_ix_x >= 0 && c_ix_x < NX)
          buff_x[dx + 1] = u[c_ix_y][c_ix_x][comp];
        else
          buff_x[dx + 1] = 0;
      }
      buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
    }
    else
      buff_y[dy + 1] = 0;
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);

  return out;
}


template <typename T>
__device__ T cuda_interpolateMatrixField2d_bicubicHermiteSpline(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u, 
                                        const int NY, const int NX,
                                        const float LY, const float LX,
                                        const float hY, const float hX,
                                        const T coord_y_warped, const T coord_x_warped,
                                        const int comp_i, const int comp_j) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY);
  const T wy = coord_y_warped / hY - iy_f;

  T buff_y[4];
  T buff_x[4];

  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_ix_y = iy_f + dy;

    if (c_ix_y >= 0 && c_ix_y < NY)
    {
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_ix_x = ix_f + dx;
        if (c_ix_x >= 0 && c_ix_x < NX)
          buff_x[dx + 1] = u[c_ix_y][c_ix_x][comp_i][comp_j];
        else
          buff_x[dx + 1] = 0;
      }
      buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
    }
    else
      buff_y[dy + 1] = 0;
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);

  return out;
}


template <typename T>
__device__ T cuda_interpolate3d_tricubicHermiteSpline(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u, 
                                         const int NZ, const int NY, const int NX,
                                         const float LZ, const float LY, const float LX,
                                         const float hZ, const float hY, const float hX,
                                         const T coord_z_warped, const T coord_y_warped, const T coord_x_warped) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY);
  const T wy = coord_y_warped / hY - iy_f;

  const int iz_f = floorf(coord_z_warped / hZ);
  const T wz = coord_z_warped / hZ - iz_f;

  T buff_z[4];
  T buff_y[4];
  T buff_x[4];

  for (int dz = -1; dz < 3; ++dz)
  {
    const int c_ix_z = iz_f + dz;
    if (c_ix_z >= 0 && c_ix_z < NZ)
    {
      for (int dy = -1; dy < 3; ++dy)
      {
        const int c_ix_y = iy_f + dy;

        if (c_ix_y >= 0 && c_ix_y < NY)
        {
          for (int dx = -1; dx < 3; ++dx)
          {
            const int c_ix_x = ix_f + dx;
            if (c_ix_x >= 0 && c_ix_x < NX)
              buff_x[dx + 1] = u[c_ix_z][c_ix_y][c_ix_x];
            else
              buff_x[dx + 1] = 0;
          }
          buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
        }
        else
          buff_y[dy + 1] = 0;
      }
      buff_z[dz + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);
    }else{
        buff_z[dz + 1] = 0;
    }
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_z, wz + 1);

  return out;
}



template <typename T>
__device__ T cuda_interpolateVectorField3d_tricubicHermiteSpline(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u, 
                                         const int NZ, const int NY, const int NX,
                                         const float LZ, const float LY, const float LX,
                                         const float hZ, const float hY, const float hX,
                                         const T coord_z_warped, const T coord_y_warped, const T coord_x_warped,
                                         const int comp) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY);
  const T wy = coord_y_warped / hY - iy_f;

  const int iz_f = floorf(coord_z_warped / hZ);
  const T wz = coord_z_warped / hZ - iz_f;

  T buff_z[4];
  T buff_y[4];
  T buff_x[4];

  for (int dz = -1; dz < 3; ++dz)
  {
    const int c_ix_z = iz_f + dz;
    if (c_ix_z >= 0 && c_ix_z < NZ)
    {
      for (int dy = -1; dy < 3; ++dy)
      {
        const int c_ix_y = iy_f + dy;

        if (c_ix_y >= 0 && c_ix_y < NY)
        {
          for (int dx = -1; dx < 3; ++dx)
          {
            const int c_ix_x = ix_f + dx;
            if (c_ix_x >= 0 && c_ix_x < NX)
              buff_x[dx + 1] = u[c_ix_z][c_ix_y][c_ix_x][comp];
            else
              buff_x[dx + 1] = 0;
          }
          buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
        }
        else
          buff_y[dy + 1] = 0;
      }
      buff_z[dz + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);
    }else{
        buff_z[dz + 1] = 0;
    }
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_z, wz + 1);

  return out;
}



template <typename T>
__device__ T cuda_interpolateMatrixField3d_tricubicHermiteSpline(const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u, 
                                         const int NZ, const int NY, const int NX,
                                         const float LZ, const float LY, const float LX,
                                         const float hZ, const float hY, const float hX,
                                         const T coord_z_warped, const T coord_y_warped, const T coord_x_warped,
                                         const int comp_i, const int comp_j) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY);
  const T wy = coord_y_warped / hY - iy_f;

  const int iz_f = floorf(coord_z_warped / hZ);
  const T wz = coord_z_warped / hZ - iz_f;

  T buff_z[4];
  T buff_y[4];
  T buff_x[4];

  for (int dz = -1; dz < 3; ++dz)
  {
    const int c_ix_z = iz_f + dz;
    if (c_ix_z >= 0 && c_ix_z < NZ)
    {
      for (int dy = -1; dy < 3; ++dy)
      {
        const int c_ix_y = iy_f + dy;

        if (c_ix_y >= 0 && c_ix_y < NY)
        {
          for (int dx = -1; dx < 3; ++dx)
          {
            const int c_ix_x = ix_f + dx;
            if (c_ix_x >= 0 && c_ix_x < NX)
              buff_x[dx + 1] = u[c_ix_z][c_ix_y][c_ix_x][comp_i][comp_j];
            else
              buff_x[dx + 1] = 0;
          }
          buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
        }
        else
          buff_y[dy + 1] = 0;
      }
      buff_z[dz + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);
    }else{
        buff_z[dz + 1] = 0;
    }
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_z, wz + 1);

  return out;
}