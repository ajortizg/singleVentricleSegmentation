#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 

#include <iostream>
#include <stdio.h>

#include "coreDefines.h"
// #include "interpolation_cubicHermiteSpline.cu"
// #include "interpolation_linear.cu"
// #include "interpolation_nearest.cu"
#include "interpolation.cuh"
#include "boundary.cuh"
#include "cudaDebug.cuh"


// CUDA kernels

//==========================================
// nearest
//==========================================

template <typename T>
__global__ void cuda_prolongate1d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const int NX, const float LX, const float hX,
  const int NX_Prolong, const float LX_Prolong, const float hX_Prolong,
  const int boundary,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < NX_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    u_prolongated[ix] = cuda_interpolate1d_nearest(u, NX, LX, hX, boundary, coord_x );
  }
}

template <typename T>
__global__ void cuda_prolongate2d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int NY_Prolong, const int NX_Prolong,
  const float LY_Prolong, const float LX_Prolong,
  const float hY_Prolong, const float hX_Prolong,
  const int boundary,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX_Prolong && iy < NY_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    u_prolongated[iy][ix] = cuda_interpolate2d_nearest(u, NY, NX, LY, LX, hY, hX, boundary, coord_y, coord_x);
  }
}


template <typename T>
__global__ void cuda_prolongateVectorField2d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int NY_Prolong, const int NX_Prolong,
  const float LY_Prolong, const float LX_Prolong,
  const float hY_Prolong, const float hX_Prolong,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX_Prolong && iy < NY_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    for(int comp=0; comp<numComp; ++comp)
    {
      u_prolongated[iy][ix][comp] = cuda_interpolateVectorField2d_nearest(u, NY, NX, LY, LX, hY, hX, boundary, coord_y, coord_x, comp);
    }
  }
}


template <typename T>
__global__ void cuda_prolongateMatrixField2d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int NY_Prolong, const int NX_Prolong,
  const float LY_Prolong, const float LX_Prolong,
  const float hY_Prolong, const float hX_Prolong,
  const int numComp_i, const int numComp_j,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX_Prolong && iy < NY_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    for(int comp_i=0; comp_i<numComp_i; ++comp_i)
      for(int comp_j=0; comp_j<numComp_j; ++comp_j)
      {
        u_prolongated[iy][ix][comp_i][comp_j] = cuda_interpolateMatrixField2d_nearest(u, NY, NX, LY, LX, hY, hX, boundary, coord_y, coord_x, comp_i, comp_j);
      }
  }
}

template <typename T>
__global__ void cuda_prolongate3d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int NZ_Prolong, const int NY_Prolong, const int NX_Prolong,
  const float LZ_Prolong, const float LY_Prolong, const float LX_Prolong,
  const float hZ_Prolong, const float hY_Prolong, const float hX_Prolong,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX_Prolong && iy < NY_Prolong && iz < NZ_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    const T coord_z_prolongated = iz * hZ_Prolong;
    const T coord_z = coord_z_prolongated * LZ / LZ_Prolong;
    u_prolongated[iz][iy][ix] = cuda_interpolate3d_nearest(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z, coord_y, coord_x);
  }
}

template <typename T>
__global__ void cuda_prolongateVectorField3d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int NZ_Prolong, const int NY_Prolong, const int NX_Prolong,
  const float LZ_Prolong, const float LY_Prolong, const float LX_Prolong,
  const float hZ_Prolong, const float hY_Prolong, const float hX_Prolong,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX_Prolong && iy < NY_Prolong && iz < NZ_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    const T coord_z_prolongated = iz * hZ_Prolong;
    const T coord_z = coord_z_prolongated * LZ / LZ_Prolong;
    for(int comp=0; comp<numComp; ++comp)
    {
      u_prolongated[iz][iy][ix][comp] = cuda_interpolateVectorField3d_nearest(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z, coord_y, coord_x, comp);
    }
  }
}


template <typename T>
__global__ void cuda_prolongateMatrixField3d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int NZ_Prolong, const int NY_Prolong, const int NX_Prolong,
  const float LZ_Prolong, const float LY_Prolong, const float LX_Prolong,
  const float hZ_Prolong, const float hY_Prolong, const float hX_Prolong,
  const int numComp_i, const int numComp_j,
  const int boundary,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX_Prolong && iy < NY_Prolong && iz < NZ_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    const T coord_z_prolongated = iz * hZ_Prolong;
    const T coord_z = coord_z_prolongated * LZ / LZ_Prolong;
    for(int comp_i=0; comp_i<numComp_i; ++comp_i)
      for(int comp_j=0; comp_j<numComp_j; ++comp_j)
      {
        u_prolongated[iz][iy][ix][comp_i][comp_j] = cuda_interpolateMatrixField3d_nearest(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z, coord_y, coord_x, comp_i, comp_j);
      }
  }
}


//==========================================
// linear
//==========================================

template <typename T>
__global__ void cuda_prolongate1d_linear_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const int NX, const float LX, const float hX,
  const int NX_Prolong, const float LX_Prolong, const float hX_Prolong,
  const int boundary,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < NX_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    u_prolongated[ix] = cuda_interpolate1d_linear(u, NX, LX, hX, boundary, coord_x );
  }
}

template <typename T>
__global__ void cuda_prolongate2d_bilinear_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int NY_Prolong, const int NX_Prolong,
  const float LY_Prolong, const float LX_Prolong,
  const float hY_Prolong, const float hX_Prolong,
  const int boundary,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX_Prolong && iy < NY_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    u_prolongated[iy][ix] = cuda_interpolate2d_bilinear(u, NY, NX, LY, LX, hY, hX, boundary, coord_y, coord_x);
  }
}


template <typename T>
__global__ void cuda_prolongateVectorField2d_bilinear_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int NY_Prolong, const int NX_Prolong,
  const float LY_Prolong, const float LX_Prolong,
  const float hY_Prolong, const float hX_Prolong,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX_Prolong && iy < NY_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    for(int comp=0; comp<2; ++comp)
    {
      u_prolongated[iy][ix][comp] = cuda_interpolateVectorField2d_bilinear(u, NY, NX, LY, LX, hY, hX, boundary, coord_y, coord_x, comp);
    }
  }
}


template <typename T>
__global__ void cuda_prolongateMatrixField2d_bilinear_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int NY_Prolong, const int NX_Prolong,
  const float LY_Prolong, const float LX_Prolong,
  const float hY_Prolong, const float hX_Prolong,
  const int numComp_i, const int numComp_j,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX_Prolong && iy < NY_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    for(int comp_i=0; comp_i<numComp_i; ++comp_i)
      for(int comp_j=0; comp_j<numComp_j; ++comp_j)
      {
        u_prolongated[iy][ix][comp_i][comp_j] = cuda_interpolateMatrixField2d_bilinear(u, NY, NX, LY, LX, hY, hX, boundary, coord_y, coord_x, comp_i, comp_j);
      }
  }
}

template <typename T>
__global__ void cuda_prolongate3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int NZ_Prolong, const int NY_Prolong, const int NX_Prolong,
  const float LZ_Prolong, const float LY_Prolong, const float LX_Prolong,
  const float hZ_Prolong, const float hY_Prolong, const float hX_Prolong,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX_Prolong && iy < NY_Prolong && iz < NZ_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    const T coord_z_prolongated = iz * hZ_Prolong;
    const T coord_z = coord_z_prolongated * LZ / LZ_Prolong;
    u_prolongated[iz][iy][ix] = cuda_interpolate3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z, coord_y, coord_x);
  }
}

template <typename T>
__global__ void cuda_prolongateVectorField3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int NZ_Prolong, const int NY_Prolong, const int NX_Prolong,
  const float LZ_Prolong, const float LY_Prolong, const float LX_Prolong,
  const float hZ_Prolong, const float hY_Prolong, const float hX_Prolong,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX_Prolong && iy < NY_Prolong && iz < NZ_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    const T coord_z_prolongated = iz * hZ_Prolong;
    const T coord_z = coord_z_prolongated * LZ / LZ_Prolong;
    for(int comp=0; comp<numComp; ++comp)
    {
      u_prolongated[iz][iy][ix][comp] = cuda_interpolateVectorField3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z, coord_y, coord_x, comp);
    }
  }
}


template <typename T>
__global__ void cuda_prolongateMatrixField3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int NZ_Prolong, const int NY_Prolong, const int NX_Prolong,
  const float LZ_Prolong, const float LY_Prolong, const float LX_Prolong,
  const float hZ_Prolong, const float hY_Prolong, const float hX_Prolong,
  const int numComp_i, const int numComp_j,
  const int boundary,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX_Prolong && iy < NY_Prolong && iz < NZ_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    const T coord_z_prolongated = iz * hZ_Prolong;
    const T coord_z = coord_z_prolongated * LZ / LZ_Prolong;
    for(int comp_i=0; comp_i<numComp_i; ++comp_i)
      for(int comp_j=0; comp_j<numComp_j; ++comp_j)
      {
        u_prolongated[iz][iy][ix][comp_i][comp_j] = cuda_interpolateMatrixField3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z, coord_y, coord_x, comp_i, comp_j);
      }
  }
}



//==========================================
// cubic Hermite spline
//==========================================

template <typename T>
__global__ void cuda_prolongate1d_cubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const int NX, const float LX, const float hX,
  const int NX_Prolong, const float LX_Prolong, const float hX_Prolong,
  const int boundary,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < NX_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    u_prolongated[ix] = cuda_interpolate1d_cubicHermiteSpline(u, NX, LX, hX, boundary, coord_x);
  }
}

template <typename T>
__global__ void cuda_prolongate2d_bicubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int NY_Prolong, const int NX_Prolong,
  const float LY_Prolong, const float LX_Prolong,
  const float hY_Prolong, const float hX_Prolong,
  const int boundary,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX_Prolong && iy < NY_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    u_prolongated[iy][ix] = cuda_interpolate2d_bicubicHermiteSpline(u, NY, NX, LY, LX, hY, hX, boundary, coord_y, coord_x);
  }
}

template <typename T>
__global__ void cuda_prolongateVectorField2d_bicubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int NY_Prolong, const int NX_Prolong,
  const float LY_Prolong, const float LX_Prolong,
  const float hY_Prolong, const float hX_Prolong,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX_Prolong && iy < NY_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    for(int comp=0; comp<numComp; ++comp)
    {
       u_prolongated[iy][ix][comp] = cuda_interpolateVectorField2d_bicubicHermiteSpline(u, NY, NX, LY, LX, hY, hX, boundary, coord_y, coord_x, comp);
    }    
  }
}

template <typename T>
__global__ void cuda_prolongateMatrixField2d_bicubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int NY_Prolong, const int NX_Prolong,
  const float LY_Prolong, const float LX_Prolong,
  const float hY_Prolong, const float hX_Prolong,
  const int numComp_i, const int numComp_j,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX_Prolong && iy < NY_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    for(int comp_i=0; comp_i<numComp_i; ++comp_i)
      for(int comp_j=0; comp_j<numComp_j; ++comp_j)
      {
        u_prolongated[iy][ix][comp_i][comp_j] = cuda_interpolateMatrixField2d_bicubicHermiteSpline(u, NY, NX, LY, LX, hY, hX, boundary, coord_y, coord_x, comp_i, comp_j );
      }
  }
}



template <typename T>
__global__ void cuda_prolongate3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int NZ_Prolong, const int NY_Prolong, const int NX_Prolong,
  const float LZ_Prolong, const float LY_Prolong, const float LX_Prolong,
  const float hZ_Prolong, const float hY_Prolong, const float hX_Prolong,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX_Prolong && iy < NY_Prolong && iz < NZ_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    const T coord_z_prolongated = iz * hZ_Prolong;
    const T coord_z = coord_z_prolongated * LZ / LZ_Prolong;
    u_prolongated[iz][iy][ix] = cuda_interpolate3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z, coord_y, coord_x);
  }
}

template <typename T>
__global__ void cuda_prolongateVectorField3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int NZ_Prolong, const int NY_Prolong, const int NX_Prolong,
  const float LZ_Prolong, const float LY_Prolong, const float LX_Prolong,
  const float hZ_Prolong, const float hY_Prolong, const float hX_Prolong,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX_Prolong && iy < NY_Prolong && iz < NZ_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    const T coord_z_prolongated = iz * hZ_Prolong;
    const T coord_z = coord_z_prolongated * LZ / LZ_Prolong;
    for(int comp=0; comp<numComp; ++comp)
    {
      u_prolongated[iz][iy][ix][comp] = cuda_interpolateVectorField3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z, coord_y, coord_x, comp);
    }
  }
}


template <typename T>
__global__ void cuda_prolongateMatrixField3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int NZ_Prolong, const int NY_Prolong, const int NX_Prolong,
  const float LZ_Prolong, const float LY_Prolong, const float LX_Prolong,
  const float hZ_Prolong, const float hY_Prolong, const float hX_Prolong,
  const int numComp_i, const int numComp_j,
  const int boundary,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX_Prolong && iy < NY_Prolong && iz < NZ_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    const T coord_y_prolongated = iy * hY_Prolong;
    const T coord_y = coord_y_prolongated * LY / LY_Prolong;
    const T coord_z_prolongated = iz * hZ_Prolong;
    const T coord_z = coord_z_prolongated * LZ / LZ_Prolong;
    for(int comp_i=0; comp_i<numComp_i; ++comp_i)
      for(int comp_j=0; comp_j<numComp_j; ++comp_j)
      {
        u_prolongated[iz][iy][ix][comp_i][comp_j] = cuda_interpolateMatrixField3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z, coord_y, coord_x, comp_i, comp_j);
      }
  }
}

// ======================================================
// C++ kernel calls
// ======================================================
torch::Tensor cuda_prolongate1d(
  const torch::Tensor &u,
  const MeshInfo1D& meshInfo,
  const MeshInfo1D& meshInfoProlongated,
  const InterpolationType interpolation, const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 1, "Expected 1d tensor");

  const int NX = meshInfo.getNX();
  const float LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  const int NX_Prolong = meshInfoProlongated.getNX();
  const float LX_Prolong = meshInfoProlongated.getLX();
  const float hX_Prolong = meshInfoProlongated.gethX();

  auto u_prolongated = torch::zeros({NX_Prolong}, u.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate1d_nearest", ([&]{
            cuda_prolongate1d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              NX, LX, hX,
              NX_Prolong, LX_Prolong, hX_Prolong,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
          }));
        cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_LINEAR:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate1d_linear", ([&]{
            cuda_prolongate1d_linear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              NX, LX, hX,
              NX_Prolong, LX_Prolong, hX_Prolong,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
          }));
        cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate1d_cubic", ([&]{
          cuda_prolongate1d_cubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NX, LX, hX,
            NX_Prolong, LX_Prolong, hX_Prolong,
            boundary,
            u_prolongated.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
    break;

  } //end switch interpolation


#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_prolongated;
}


torch::Tensor cuda_prolongate2d(
  const torch::Tensor &u,
  const MeshInfo2D& meshInfo,
  const MeshInfo2D& meshInfoProlongated,
  const InterpolationType interpolation, const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 2, "Expected 2d tensor");

  const int NY = u.size(0);
  const int NX = u.size(1);
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  const int NY_Prolong = meshInfoProlongated.getNY();
  const int NX_Prolong = meshInfoProlongated.getNX();
  const float LY_Prolong = meshInfoProlongated.getLY();
  const float LX_Prolong = meshInfoProlongated.getLX();
  const float hY_Prolong = meshInfoProlongated.gethY();
  const float hX_Prolong = meshInfoProlongated.gethX();

  auto u_prolongated = torch::zeros({NY_Prolong,NX_Prolong}, u.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x, (NY_Prolong + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {

    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate2d_nearest", ([&]{
            cuda_prolongate2d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              NY_Prolong, NX_Prolong,
              LY_Prolong, LX_Prolong,
              hY_Prolong, hX_Prolong,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate2d_bilinear", ([&]{
            cuda_prolongate2d_bilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              NY_Prolong, NX_Prolong,
              LY_Prolong, LX_Prolong,
              hY_Prolong, hX_Prolong,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

  case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate2d_bicubic", ([&]{
          cuda_prolongate2d_bicubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              NY_Prolong, NX_Prolong,
              LY_Prolong, LX_Prolong,
              hY_Prolong, hX_Prolong,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
      break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_prolongated;
}

torch::Tensor cuda_prolongateVectorField2d(
  const torch::Tensor &u,
  const MeshInfo2D& meshInfo,
  const MeshInfo2D& meshInfoProlongated,
  const InterpolationType interpolation, const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 3, "Expected 3d tensor");

  const int NY = u.size(0);
  const int NX = u.size(1);
  const int numComp = u.size(2);
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  const int NY_Prolong = meshInfoProlongated.getNY();
  const int NX_Prolong = meshInfoProlongated.getNX();
  const float LY_Prolong = meshInfoProlongated.getLY();
  const float LX_Prolong = meshInfoProlongated.getLX();
  const float hY_Prolong = meshInfoProlongated.gethY();
  const float hX_Prolong = meshInfoProlongated.gethX();

  auto u_prolongated = torch::zeros({NY_Prolong,NX_Prolong,numComp}, u.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x, (NY_Prolong + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

 switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateVectorField2d_nearest", ([&]{
            cuda_prolongateVectorField2d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              NY_Prolong, NX_Prolong,
              LY_Prolong, LX_Prolong,
              hY_Prolong, hX_Prolong,
              numComp,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateVectorField2d_bilinear", ([&]{
            cuda_prolongateVectorField2d_bilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              NY_Prolong, NX_Prolong,
              LY_Prolong, LX_Prolong,
              hY_Prolong, hX_Prolong,
              numComp,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;
        
  case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateVectorField2d_bicubic", ([&]{
          cuda_prolongateVectorField2d_bicubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              NY_Prolong, NX_Prolong,
              LY_Prolong, LX_Prolong,
              hY_Prolong, hX_Prolong,
              numComp,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
      break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_prolongated;
}

torch::Tensor cuda_prolongateMatrixField2d(
  const torch::Tensor &u,
  const MeshInfo2D& meshInfo,
  const MeshInfo2D& meshInfoProlongated,
  const InterpolationType interpolation, const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 4, "Expected 4d tensor");

  const int NY = u.size(0);
  const int NX = u.size(1);
  const int numComp_i = u.size(2);
  const int numComp_j = u.size(3);
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  const int NY_Prolong = meshInfoProlongated.getNY();
  const int NX_Prolong = meshInfoProlongated.getNX();
  const float LY_Prolong = meshInfoProlongated.getLY();
  const float LX_Prolong = meshInfoProlongated.getLX();
  const float hY_Prolong = meshInfoProlongated.gethY();
  const float hX_Prolong = meshInfoProlongated.gethX();

  auto u_prolongated = torch::zeros({NY_Prolong,NX_Prolong,numComp_i,numComp_j}, u.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x, (NY_Prolong + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

 switch(interpolation)
  {

    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateMatrixField2d_nearest", ([&]{
            cuda_prolongateMatrixField2d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              NY_Prolong, NX_Prolong,
              LY_Prolong, LX_Prolong,
              hY_Prolong, hX_Prolong,
              numComp_i, numComp_j,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateMatrixField2d_bilinear", ([&]{
            cuda_prolongateMatrixField2d_bilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              NY_Prolong, NX_Prolong,
              LY_Prolong, LX_Prolong,
              hY_Prolong, hX_Prolong,
              numComp_i, numComp_j,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

  case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateMatrixField2d_bicubic", ([&]{
          cuda_prolongateMatrixField2d_bicubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              NY_Prolong, NX_Prolong,
              LY_Prolong, LX_Prolong,
              hY_Prolong, hX_Prolong,
              numComp_i, numComp_j,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
      break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_prolongated;
}


torch::Tensor cuda_prolongate3d(
  const torch::Tensor &u,
  const MeshInfo3D& meshInfo,
  const MeshInfo3D& meshInfoProlongated,
  const InterpolationType interpolation, const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 3, "Expected 3d tensor");

  const int NZ = u.size(0);
  const int NY = u.size(1);
  const int NX = u.size(2);
  const float LZ = meshInfo.getLZ();
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  const int NZ_Prolong = meshInfoProlongated.getNZ();
  const int NY_Prolong = meshInfoProlongated.getNY();
  const int NX_Prolong = meshInfoProlongated.getNX();
  const float LZ_Prolong = meshInfoProlongated.getLZ();
  const float LY_Prolong = meshInfoProlongated.getLY();
  const float LX_Prolong = meshInfoProlongated.getLX();
  const float hZ_Prolong = meshInfoProlongated.gethZ();
  const float hY_Prolong = meshInfoProlongated.gethY();
  const float hX_Prolong = meshInfoProlongated.gethX();

  auto u_prolongated = torch::zeros({NZ_Prolong,NY_Prolong,NX_Prolong}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x, (NY_Prolong + blockSize.y - 1) / blockSize.y, (NZ_Prolong + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate3d_nearest", ([&]{
            cuda_prolongate3d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NZ, NY, NX,
              LZ, LY, LX,
              hZ, hY, hX,
              NZ_Prolong, NY_Prolong, NX_Prolong,
              LZ_Prolong, LY_Prolong, LX_Prolong,
              hZ_Prolong, hY_Prolong, hX_Prolong,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate3d_trilinear", ([&]{
            cuda_prolongate3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NZ, NY, NX,
              LZ, LY, LX,
              hZ, hY, hX,
              NZ_Prolong, NY_Prolong, NX_Prolong,
              LZ_Prolong, LY_Prolong, LX_Prolong,
              hZ_Prolong, hY_Prolong, hX_Prolong,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

  case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate3d_tricubic", ([&]{
          cuda_prolongate3d_tricubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NZ, NY, NX,
              LZ, LY, LX,
              hZ, hY, hX,
              NZ_Prolong, NY_Prolong, NX_Prolong,
              LZ_Prolong, LY_Prolong, LX_Prolong,
              hZ_Prolong, hY_Prolong, hX_Prolong,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
      break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_prolongated;
}



torch::Tensor cuda_prolongateVectorField3d(
  const torch::Tensor &u,
  const MeshInfo3D& meshInfo,
  const MeshInfo3D& meshInfoProlongated,
  const InterpolationType interpolation, const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 4, "Expected 4d tensor");

  const int NZ = u.size(0);
  const int NY = u.size(1);
  const int NX = u.size(2);
  const int numComp = u.size(3);
  const float LZ = meshInfo.getLZ();
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  const int NZ_Prolong = meshInfoProlongated.getNZ();
  const int NY_Prolong = meshInfoProlongated.getNY();
  const int NX_Prolong = meshInfoProlongated.getNX();
  const float LZ_Prolong = meshInfoProlongated.getLZ();
  const float LY_Prolong = meshInfoProlongated.getLY();
  const float LX_Prolong = meshInfoProlongated.getLX();
  const float hZ_Prolong = meshInfoProlongated.gethZ();
  const float hY_Prolong = meshInfoProlongated.gethY();
  const float hX_Prolong = meshInfoProlongated.gethX();

  auto u_prolongated = torch::zeros({NZ_Prolong,NY_Prolong,NX_Prolong,numComp}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x, (NY_Prolong + blockSize.y - 1) / blockSize.y, (NZ_Prolong + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateVectorField3d_nearest", ([&]{
            cuda_prolongateVectorField3d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              NZ, NY, NX,
              LZ, LY, LX,
              hZ, hY, hX,
              NZ_Prolong, NY_Prolong, NX_Prolong,
              LZ_Prolong, LY_Prolong, LX_Prolong,
              hZ_Prolong, hY_Prolong, hX_Prolong,
              numComp,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateVectorField3d_trilinear", ([&]{
            cuda_prolongateVectorField3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              NZ, NY, NX,
              LZ, LY, LX,
              hZ, hY, hX,
              NZ_Prolong, NY_Prolong, NX_Prolong,
              LZ_Prolong, LY_Prolong, LX_Prolong,
              hZ_Prolong, hY_Prolong, hX_Prolong,
              numComp,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

  case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateVectorField3d_tricubic", ([&]{
          cuda_prolongateVectorField3d_tricubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              NZ, NY, NX,
              LZ, LY, LX,
              hZ, hY, hX,
              NZ_Prolong, NY_Prolong, NX_Prolong,
              LZ_Prolong, LY_Prolong, LX_Prolong,
              hZ_Prolong, hY_Prolong, hX_Prolong,
              numComp,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
      break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_prolongated;
}



torch::Tensor cuda_prolongateMatrixField3d(
  const torch::Tensor &u,
  const MeshInfo3D& meshInfo,
  const MeshInfo3D& meshInfoProlongated,
  const InterpolationType interpolation, const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 5, "Expected 5d tensor");

  const int NZ = u.size(0);
  const int NY = u.size(1);
  const int NX = u.size(2);
  const int numComp_i = u.size(3);
  const int numComp_j = u.size(4);
  const float LZ = meshInfo.getLZ();
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  const int NZ_Prolong = meshInfoProlongated.getNZ();
  const int NY_Prolong = meshInfoProlongated.getNY();
  const int NX_Prolong = meshInfoProlongated.getNX();
  const float LZ_Prolong = meshInfoProlongated.getLZ();
  const float LY_Prolong = meshInfoProlongated.getLY();
  const float LX_Prolong = meshInfoProlongated.getLX();
  const float hZ_Prolong = meshInfoProlongated.gethZ();
  const float hY_Prolong = meshInfoProlongated.gethY();
  const float hX_Prolong = meshInfoProlongated.gethX();

  auto u_prolongated = torch::zeros({NZ_Prolong,NY_Prolong,NX_Prolong,numComp_i,numComp_j}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x, (NY_Prolong + blockSize.y - 1) / blockSize.y, (NZ_Prolong + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateMatrixField3d_nearest", ([&]{
            cuda_prolongateMatrixField3d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
              NZ, NY, NX,
              LZ, LY, LX,
              hZ, hY, hX,
              NZ_Prolong, NY_Prolong, NX_Prolong,
              LZ_Prolong, LY_Prolong, LX_Prolong,
              hZ_Prolong, hY_Prolong, hX_Prolong,
              numComp_i, numComp_j,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateMatrixField3d_trilinear", ([&]{
            cuda_prolongateMatrixField3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
              NZ, NY, NX,
              LZ, LY, LX,
              hZ, hY, hX,
              NZ_Prolong, NY_Prolong, NX_Prolong,
              LZ_Prolong, LY_Prolong, LX_Prolong,
              hZ_Prolong, hY_Prolong, hX_Prolong,
              numComp_i, numComp_j,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
        break;

  case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongateMatrixField3d_tricubic", ([&]{
          cuda_prolongateMatrixField3d_tricubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
              NZ, NY, NX,
              LZ, LY, LX,
              hZ, hY, hX,
              NZ_Prolong, NY_Prolong, NX_Prolong,
              LZ_Prolong, LY_Prolong, LX_Prolong,
              hZ_Prolong, hY_Prolong, hX_Prolong,
              numComp_i, numComp_j,
              boundary,
              u_prolongated.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
      break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_prolongated;
}