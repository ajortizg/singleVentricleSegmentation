#ifndef __INTERPOLATION_CUBICHS_CUH_
#define __INTERPOLATION_CUBICHS_CUH_

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
// local cubic hermite spline interpolation
//=========================================================

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


//=========================================================
// cubic hermite spline interpolation in 1D
//=========================================================

template <typename T>
__device__ T cuda_interpolate1d_cubicHermiteSpline(
    const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
    const int NX, const float LX, const float hX, const int boundary, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;
  T buff_x[4];
  for (int dx = -1; dx < 3; ++dx)
  {
      const int c_ix_x = ix_f + dx;
      const int c_ix_x_out = getIndexInterpolate(c_ix_x,NX,boundary);
      buff_x[dx + 1] = u[c_ix_x_out];
  }
  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
  return out;
}


//=========================================================
// bi-cubic hermite spline interpolation in 2D
//=========================================================

//=====================
// scalar fields
//=====================
template <typename T>
__device__ T cuda_interpolate2d_bicubicHermiteSpline(const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u, 
                                        const int NY, const int NX,
                                        const float LY, const float LX,
                                        const float hY, const float hX,
                                        const int boundary,
                                        const T coord_y_warped, const T coord_x_warped) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;
  T buff_x[4];

  const int iy_f = floorf(coord_y_warped / hY);
  const T wy = coord_y_warped / hY - iy_f;
  T buff_y[4];

  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_ix_y = iy_f + dy;
    const int c_ix_y_out = getIndexInterpolate(c_ix_y,NY,boundary);
    for (int dx = -1; dx < 3; ++dx)
    {
        const int c_ix_x = ix_f + dx;
        const int c_ix_x_out = getIndexInterpolate(c_ix_x,NX,boundary);
        buff_x[dx + 1] = u[c_ix_y_out][c_ix_x_out];
    }
    buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);

  return out;
}


//=====================
// vector fields
//=====================
template <typename T>
__device__ T cuda_interpolateVectorField2d_bicubicHermiteSpline( const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u, 
                                        const int NY, const int NX,
                                        const float LY, const float LX,
                                        const float hY, const float hX,
                                        const int boundary,
                                        const T coord_y_warped, const T coord_x_warped,
                                        const int comp ) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;
  T buff_x[4];

  const int iy_f = floorf(coord_y_warped / hY);
  const T wy = coord_y_warped / hY - iy_f;
  T buff_y[4];

  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_ix_y = iy_f + dy;
    const int c_ix_y_out = getIndexInterpolate(c_ix_y,NY,boundary);
    for (int dx = -1; dx < 3; ++dx)
    {
        const int c_ix_x = ix_f + dx;
        const int c_ix_x_out = getIndexInterpolate(c_ix_x,NX,boundary);
        buff_x[dx + 1] = u[c_ix_y_out][c_ix_x_out][comp];
    }
    buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);

  return out;
}

//=====================
// matrix fields
//=====================
template <typename T>
__device__ T cuda_interpolateMatrixField2d_bicubicHermiteSpline(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u, 
                                        const int NY, const int NX,
                                        const float LY, const float LX,
                                        const float hY, const float hX,
                                        const int boundary,
                                        const T coord_y_warped, const T coord_x_warped,
                                        const int comp_i, const int comp_j ) {

  const int ix_f = floorf(coord_x_warped / hX);
  const T wx = coord_x_warped / hX - ix_f;
  T buff_x[4];

  const int iy_f = floorf(coord_y_warped / hY);
  const T wy = coord_y_warped / hY - iy_f;
  T buff_y[4];

  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_ix_y = iy_f + dy;
    const int c_ix_y_out = getIndexInterpolate(c_ix_y,NY,boundary);
    for (int dx = -1; dx < 3; ++dx)
    {
        const int c_ix_x = ix_f + dx;
        const int c_ix_x_out = getIndexInterpolate(c_ix_x,NX,boundary);
        buff_x[dx + 1] = u[c_ix_y_out][c_ix_x_out][comp_i][comp_j];
    }
    buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);

  return out;
}

//=========================================================
// tri-cubic hermite spline interpolation in 1D
//=========================================================

//=====================
// scalar fields
//=====================
template <typename T>
__device__ T cuda_interpolate3d_tricubicHermiteSpline(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u, 
                                         const int NZ, const int NY, const int NX,
                                         const float LZ, const float LY, const float LX,
                                         const float hZ, const float hY, const float hX,
                                         const int boundary,
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
    const int c_ix_z_out = getIndexInterpolate(c_ix_z,NZ,boundary);
    for (int dy = -1; dy < 3; ++dy)
    {
      const int c_ix_y = iy_f + dy;
      const int c_ix_y_out = getIndexInterpolate(c_ix_y,NY,boundary);
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_ix_x = ix_f + dx;
        const int c_ix_x_out = getIndexInterpolate(c_ix_x,NX,boundary);
        buff_x[dx + 1] = u[c_ix_z_out][c_ix_y_out][c_ix_x_out];
      }
      buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
    }
    buff_z[dz + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_z, wz + 1);

  return out;
}

//=====================
// vector fields
//=====================
template <typename T>
__device__ T cuda_interpolateVectorField3d_tricubicHermiteSpline(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u, 
                                         const int NZ, const int NY, const int NX,
                                         const float LZ, const float LY, const float LX,
                                         const float hZ, const float hY, const float hX,
                                         const int boundary,
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
    const int c_ix_z_out = getIndexInterpolate(c_ix_z,NZ,boundary);
    for (int dy = -1; dy < 3; ++dy)
    {
      const int c_ix_y = iy_f + dy;
      const int c_ix_y_out = getIndexInterpolate(c_ix_y,NY,boundary);
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_ix_x = ix_f + dx;
        const int c_ix_x_out = getIndexInterpolate(c_ix_x,NX,boundary);
        buff_x[dx + 1] = u[c_ix_z_out][c_ix_y_out][c_ix_x_out][comp];
      }
      buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
    }
    buff_z[dz + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_z, wz + 1);

  return out;
}


//=====================
// matrix fields
//=====================
template <typename T>
__device__ T cuda_interpolateMatrixField3d_tricubicHermiteSpline(const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u, 
                                         const int NZ, const int NY, const int NX,
                                         const float LZ, const float LY, const float LX,
                                         const float hZ, const float hY, const float hX,
                                         const int boundary,
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
    const int c_ix_z_out = getIndexInterpolate(c_ix_z,NZ,boundary);
    for (int dy = -1; dy < 3; ++dy)
    {
      const int c_ix_y = iy_f + dy;
      const int c_ix_y_out = getIndexInterpolate(c_ix_y,NY,boundary);
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_ix_x = ix_f + dx;
        const int c_ix_x_out = getIndexInterpolate(c_ix_x,NX,boundary);
        buff_x[dx + 1] = u[c_ix_z_out][c_ix_y_out][c_ix_x_out][comp_i][comp_j];
      }
      buff_y[dy + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_x, wx + 1);
    }
    buff_z[dz + 1] = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_y, wy + 1);
  }

  T out = cuda_interpolate1d_cubicHermiteSpline_local<T>(buff_z, wz + 1);

  return out;
}

#endif