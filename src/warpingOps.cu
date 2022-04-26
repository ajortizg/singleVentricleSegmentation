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
__global__ void cuda_warp1d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> phi,
  const int NX, const float LX, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < NX )
  {
    const T dx = phi[ix][0];
    const T coord_x_warped = ix * hX + dx;
    u_warped[ix] = cuda_interpolate1d_nearest(u, NX, LX, hX, boundary, coord_x_warped );
  }  
}

template <typename T>
__global__ void cuda_warp1d_nearest_backward_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> phi,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> forward_out,
  const int NX, const float LX, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> grad_u,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_phi )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < NX )
  {
    const T dx = phi[ix][0];
    const T coord_x_warped = ix * hX + dx;
    const T forward_val = forward_out[ix];
    T grad_phi_idx = 0;
    cuda_interpolate1d_nearest_backward(u, NX, LX, hX, boundary, coord_x_warped, forward_val, grad_u, grad_phi_idx );
    grad_phi[ix][0] = grad_phi_idx;
  }  
}


template <typename T>
__global__ void cuda_warp2d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    u_warped[iy][ix] = cuda_interpolate2d_nearest(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped);
  }
}

template <typename T>
__global__ void cuda_warp2d_nearest_backward_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> forward_out,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_u,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_phi )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T forward_val = forward_out[iy][ix];
    T grad_phi_idy = 0; T grad_phi_idx = 0;
    cuda_interpolate2d_nearest_backward(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped, forward_val, grad_u, grad_phi_idy, grad_phi_idx );
    grad_phi[iy][ix][0] = grad_phi_idx;
    grad_phi[iy][ix][1] = grad_phi_idy;
  }  
}

template <typename T>
__global__ void cuda_warpVectorField2d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    for(int comp=0; comp<numComp; ++comp)
    {
      u_warped[iy][ix][comp] = cuda_interpolateVectorField2d_nearest(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped, comp);
    }
  }
}

template <typename T>
__global__ void cuda_warp3d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T dx = phi[iz][iy][ix][0];
    const T dy = phi[iz][iy][ix][1];
    const T dz = phi[iz][iy][ix][2];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T coord_z_warped = iz * hZ + dz;
    u_warped[iz][iy][ix] = cuda_interpolate3d_nearest(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped);
  }
}

template <typename T>
__global__ void cuda_warp3d_nearest_backward_kernel(
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
    const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> forward_out,
    const int NZ, const int NY, const int NX,
    const float LZ, const float LY, const float LX,
    const float hZ, const float hY, const float hX,
    const int boundary,
    torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_u,
    torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> grad_phi )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T dx = phi[iz][iy][ix][0];
    const T dy = phi[iz][iy][ix][1];
    const T dz = phi[iz][iy][ix][2];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T coord_z_warped = iz * hZ + dz;
    const T forward_val = forward_out[iz][iy][ix];
    T grad_phi_idz = 0; T grad_phi_idy = 0; T grad_phi_idx = 0;
    cuda_interpolate3d_nearest_backward(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped, forward_val, grad_u, grad_phi_idz, grad_phi_idy, grad_phi_idx );
    grad_phi[iz][iy][ix][0] = grad_phi_idx;
    grad_phi[iz][iy][ix][1] = grad_phi_idy;
    grad_phi[iz][iy][ix][2] = grad_phi_idz;
  }  
}

template <typename T>
__global__ void cuda_warpVectorField3d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T dx = phi[iz][iy][ix][0];
    const T dy = phi[iz][iy][ix][1];
    const T dz = phi[iz][iy][ix][2];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T coord_z_warped = iz * hZ + dz;
    for(int comp=0; comp<numComp; ++comp)
    {
      u_warped[iz][iy][ix][comp] = cuda_interpolateVectorField3d_nearest(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped, comp);
    }
  }
}

//==========================================
// linear
//==========================================

template <typename T>
__global__ void cuda_warp1d_linear_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> phi,
  const int NX, const float LX, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < NX )
  {
    const T dx = phi[ix][0];
    const T coord_x_warped = ix * hX + dx;
    u_warped[ix] = cuda_interpolate1d_linear(u, NX, LX, hX, boundary, coord_x_warped );
  }  
}

template <typename T>
__global__ void cuda_warp1d_linear_backward_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> phi,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> forward_out,
  const int NX, const float LX, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> grad_u,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_phi )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < NX )
  {
    const T dx = phi[ix][0];
    const T coord_x_warped = ix * hX + dx;
    const T forward_val = forward_out[ix];
    T grad_phi_idx = 0;
    cuda_interpolate1d_linear_backward(u, NX, LX, hX, boundary, coord_x_warped, forward_val, grad_u, grad_phi_idx );
    grad_phi[ix][0] = grad_phi_idx;
  }  
}


template <typename T>
__global__ void cuda_warp2d_bilinear_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    u_warped[iy][ix] = cuda_interpolate2d_bilinear(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped);
  }
}

template <typename T>
__global__ void cuda_warp2d_bilinear_backward_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> forward_out,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_u,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_phi )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T forward_val = forward_out[iy][ix];
    T grad_phi_idy = 0; T grad_phi_idx = 0;
    cuda_interpolate2d_bilinear_backward(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped, forward_val, grad_u, grad_phi_idy, grad_phi_idx );
    grad_phi[iy][ix][0] = grad_phi_idx;
    grad_phi[iy][ix][1] = grad_phi_idy;
  }  
}

template <typename T>
__global__ void cuda_warpVectorField2d_bilinear_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    for(int comp=0; comp<numComp; ++comp)
    {
      u_warped[iy][ix][comp] = cuda_interpolateVectorField2d_bilinear(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped, comp);
    }
  }
}

template <typename T>
__global__ void cuda_warp3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T dx = phi[iz][iy][ix][0];
    const T dy = phi[iz][iy][ix][1];
    const T dz = phi[iz][iy][ix][2];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T coord_z_warped = iz * hZ + dz;
    u_warped[iz][iy][ix] = cuda_interpolate3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped);
  }
}

template <typename T>
__global__ void cuda_warp3d_trilinear_backward_kernel(
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
    const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> forward_out,
    const int NZ, const int NY, const int NX,
    const float LZ, const float LY, const float LX,
    const float hZ, const float hY, const float hX,
    const int boundary,
    torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_u,
    torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> grad_phi )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T dx = phi[iz][iy][ix][0];
    const T dy = phi[iz][iy][ix][1];
    const T dz = phi[iz][iy][ix][2];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T coord_z_warped = iz * hZ + dz;
    const T forward_val = forward_out[iz][iy][ix];
    T grad_phi_idz = 0; T grad_phi_idy = 0; T grad_phi_idx = 0;
    cuda_interpolate3d_trilinear_backward(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped, forward_val, grad_u, grad_phi_idz, grad_phi_idy, grad_phi_idx );
    grad_phi[iz][iy][ix][0] = grad_phi_idx;
    grad_phi[iz][iy][ix][1] = grad_phi_idy;
    grad_phi[iz][iy][ix][2] = grad_phi_idz;
  }  
}

template <typename T>
__global__ void cuda_warpVectorField3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T dx = phi[iz][iy][ix][0];
    const T dy = phi[iz][iy][ix][1];
    const T dz = phi[iz][iy][ix][2];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T coord_z_warped = iz * hZ + dz;
    for(int comp=0; comp<numComp; ++comp)
    {
      u_warped[iz][iy][ix][comp] = cuda_interpolateVectorField3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped, comp);
    }
  }
}

//==========================================
// cubic hermite spline
//==========================================

template <typename T>
__global__ void cuda_warp1d_cubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> phi,
  const int NX, const float LX, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < NX )
  {
    const T dx = phi[ix][0];
    const T coord_x_warped = ix * hX + dx;
    u_warped[ix] = cuda_interpolate1d_cubicHermiteSpline(u, NX, LX, hX, boundary, coord_x_warped);
  }
}

template <typename T>
__global__ void cuda_warp1d_cubicHermiteSpline_backward_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> phi,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> forward_out,
  const int NX, const float LX, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> grad_u,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_phi )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < NX )
  {
    const T dx = phi[ix][0];
    const T coord_x_warped = ix * hX + dx;
    const T forward_val = forward_out[ix];
    // T grad_phi_idx = 0;
    // cuda_interpolate1d_cubicHermiteSpline_backward(u, NX, LX, hX, boundary, coord_x_warped, forward_val, grad_u, grad_phi_idx );
    // grad_phi[ix][0] = grad_phi_idx;
    cuda_interpolate1d_cubicHermiteSpline_backward(u, NX, LX, hX, boundary, coord_x_warped, forward_val, ix, grad_u, grad_phi );
  }  
}

template <typename T>
__global__ void cuda_warp2d_bicubicHermiteSpline_kernel(
    const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
    const int NY, const int NX,
    const float LY, const float LX,
    const float hY, const float hX,
    const int boundary,
    torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    u_warped[iy][ix] = cuda_interpolate2d_bicubicHermiteSpline(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped);
  }
}

template <typename T>
__global__ void cuda_warp2d_bicubicHermiteSpline_backward_kernel(
    const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
    const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> forward_out,
    const int NY, const int NX,
    const float LY, const float LX,
    const float hY, const float hX,
    const int boundary,
    torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_u,
    torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_phi )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T forward_val = forward_out[iy][ix];
    // T grad_phi_idy = 0; T grad_phi_idx = 0;
    // cuda_interpolate2d_bicubicHermiteSpline_backward(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped, forward_val, grad_u, grad_phi_idy, grad_phi_idx );
    // grad_phi[iy][ix][0] = grad_phi_idx;
    // grad_phi[iy][ix][1] = grad_phi_idy;
    //Variante 2:
    cuda_interpolate2d_bicubicHermiteSpline_backward(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped, forward_val, iy, ix, grad_u, grad_phi );
  }  
}


template <typename T>
__global__ void cuda_warpVectorField2d_bicubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    for(int comp=0; comp<numComp; ++comp)
    {
      u_warped[iy][ix][comp] = cuda_interpolateVectorField2d_bicubicHermiteSpline(u, NY, NX, LY, LX, hY, hX, boundary, coord_y_warped, coord_x_warped, comp);
    }
  }
}

template <typename T>
__global__ void cuda_warp3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T dx = phi[iz][iy][ix][0];
    const T dy = phi[iz][iy][ix][1];
    const T dz = phi[iz][iy][ix][2];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T coord_z_warped = iz * hZ + dz;
    u_warped[iz][iy][ix] = cuda_interpolate3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped);
  }
}

template <typename T>
__global__ void cuda_warp3d_tricubicHermiteSpline_backward_kernel(
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
    const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
    const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> forward_out,
    const int NZ, const int NY, const int NX,
    const float LZ, const float LY, const float LX,
    const float hZ, const float hY, const float hX,
    const int boundary,
    torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> grad_u,
    torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> grad_phi )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T dx = phi[iz][iy][ix][0];
    const T dy = phi[iz][iy][ix][1];
    const T dz = phi[iz][iy][ix][2];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T coord_z_warped = iz * hZ + dz;
    const T forward_val = forward_out[iz][iy][ix];
    // T grad_phi_idz = 0; T grad_phi_idy = 0; T grad_phi_idx = 0;
    // cuda_interpolate3d_tricubicHermiteSpline_backward(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped, forward_val, grad_u, grad_phi_idz, grad_phi_idy, grad_phi_idx );
    // grad_phi[iz][iy][ix][0] = grad_phi_idx;
    // grad_phi[iz][iy][ix][1] = grad_phi_idy;
    // grad_phi[iz][iy][ix][2] = grad_phi_idz;
    cuda_interpolate3d_tricubicHermiteSpline_backward<T>(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped, forward_val, iz, iy, ix, grad_u, grad_phi );
  }  
}

template <typename T>
__global__ void cuda_warpVectorField3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T dx = phi[iz][iy][ix][0];
    const T dy = phi[iz][iy][ix][1];
    const T dz = phi[iz][iy][ix][2];
    const T coord_x_warped = ix * hX + dx;
    const T coord_y_warped = iy * hY + dy;
    const T coord_z_warped = iz * hZ + dz;
    for(int comp=0; comp<numComp; ++comp )
    {
      u_warped[iz][iy][ix][comp] = cuda_interpolateVectorField3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped, comp);
    }
  }
}

// ======================================================
// C++ kernel calls
// ======================================================
torch::Tensor cuda_warp1d(
  const torch::Tensor &u,
  const torch::Tensor &phi,
  const MeshInfo1D& meshInfo,
  const InterpolationType interpolation,
  const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 1, "Expected 1d tensor");
  TORCH_CHECK(phi.dim() == 2, "Expected 2d tensor")

  const int NX = u.size(0);
  const float LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  auto u_warped = torch::zeros({NX}, u.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

switch(interpolation)
{
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp1d_nearest", ([&]{
            cuda_warp1d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              NX, LX, hX, 
              boundary,
              u_warped.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp1d_linear", ([&]{
            cuda_warp1d_linear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              NX, LX, hX, 
              boundary,
              u_warped.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp1d_cubic", ([&]{
            cuda_warp1d_cubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              NX, LX, hX,
              boundary,
              u_warped.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
    break;

} //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}


std::vector<torch::Tensor> cuda_warp1d_backward( 
  const torch::Tensor u, 
  const torch::Tensor phi, 
  const torch::Tensor forward_out, 
  const MeshInfo1D& meshInfo, 
  const InterpolationType interpolation,
  const BoundaryType boundary)
{
  TORCH_CHECK(u.dim() == 1, "Expected 1d tensor");
  TORCH_CHECK(phi.dim() == 2, "Expected 2d tensor")
  TORCH_CHECK(forward_out.dim() == 1, "Expected 1d tensor")

  const int NX = u.size(0);
  const float LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  auto u_grad = torch::zeros_like(u);
  auto phi_grad = torch::zeros_like(phi);

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

switch(interpolation)
{
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp1d_nearest_backward", ([&]{
            cuda_warp1d_nearest_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              forward_out.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              NX, LX, hX, 
              boundary,
              u_grad.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              phi_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
              );
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp1d_linear_backward", ([&]{
            cuda_warp1d_linear_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              forward_out.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              NX, LX, hX, 
              boundary,
              u_grad.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              phi_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
              );
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp1d_cubic_backward", ([&]{
            cuda_warp1d_cubicHermiteSpline_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              forward_out.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              NX, LX, hX, 
              boundary,
              u_grad.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
              phi_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
              );
          }));
          cudaSafeCall(cudaGetLastError());
    break;

} //end switch interpolation


#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "backward time " << cut.elapsed() << std::endl;
#endif

  return {u_grad, phi_grad};
}




torch::Tensor cuda_warp2d(
  const torch::Tensor &u,
  const torch::Tensor &phi,
  const MeshInfo2D& meshInfo,
  const InterpolationType interpolation,
  const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 2, "Expected 2d tensor");
  TORCH_CHECK(phi.dim() == 3, "Expected 3d tensor")

  const int NY = u.size(0);
  const int NX = u.size(1);
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto u_warped = torch::zeros({NY,NX}, u.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp2d_nearest", ([&]{
            cuda_warp2d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              boundary,
              u_warped.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp2d_bilinear", ([&]{
            cuda_warp2d_bilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              boundary,
              u_warped.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warp2d_bicubic", ([&]{
          cuda_warp2d_bicubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            NY, NX,
            LY, LX,
            hY, hX,
            boundary,
            u_warped.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
      break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}


std::vector<torch::Tensor> cuda_warp2d_backward( 
  const torch::Tensor u, 
  const torch::Tensor phi, 
  const torch::Tensor forward_out, 
  const MeshInfo2D& meshInfo, 
  const InterpolationType interpolation,
  const BoundaryType boundary)
{
  TORCH_CHECK(u.dim() == 2, "Expected 2d tensor");
  TORCH_CHECK(phi.dim() == 3, "Expected 3d tensor")
  TORCH_CHECK(forward_out.dim() == 2, "Expected 2d tensor")

  const int NY = u.size(0);
  const int NX = u.size(1);
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto u_grad = torch::zeros_like(u);
  auto phi_grad = torch::zeros_like(phi);

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

switch(interpolation)
{
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp2d_nearest_backward", ([&]{
            cuda_warp2d_nearest_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              forward_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              NY, NX, LY, LX, hY, hX, 
              boundary,
              u_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              phi_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
              );
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp2d_bilinear_backward", ([&]{
            cuda_warp2d_bilinear_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              forward_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              NY, NX, LY, LX, hY, hX, 
              boundary,
              u_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              phi_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
              );
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp2d_bicubic_backward", ([&]{
            cuda_warp2d_bicubicHermiteSpline_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              forward_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              NY, NX, LY, LX, hY, hX, 
              boundary,
              u_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
              phi_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
              );
          }));
          cudaSafeCall(cudaGetLastError());
    break;

} //end switch interpolation


#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "backward time " << cut.elapsed() << std::endl;
#endif

  return {u_grad, phi_grad};
}



torch::Tensor cuda_warpVectorField2d(
  const torch::Tensor &u,
  const torch::Tensor &phi,
  const MeshInfo2D& meshInfo,
  const InterpolationType interpolation,
  const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 3, "Expected 3d tensor");
  TORCH_CHECK(phi.dim() == 3, "Expected 3d tensor")

  const int NY = u.size(0);
  const int NX = u.size(1);
  const int numComp = u.size(2);
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto u_warped = torch::zeros({NY,NX,numComp}, u.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warpVectorField2d_nearest", ([&]{
            cuda_warpVectorField2d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              numComp,
              boundary,
              u_warped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warpVectorField2d_bilinear", ([&]{
            cuda_warpVectorField2d_bilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NY, NX,
              LY, LX,
              hY, hX,
              numComp,
              boundary,
              u_warped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warpVectorField2d_bicubic", ([&]{
          cuda_warpVectorField2d_bicubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            NY, NX,
            LY, LX,
            hY, hX,
            numComp,
            boundary,
            u_warped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
    break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}




torch::Tensor cuda_warp3d(
  const torch::Tensor &u,
  const torch::Tensor &phi,
  const MeshInfo3D& meshInfo,
  const InterpolationType interpolation,
  const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 3, "Expected 3d tensor");
  TORCH_CHECK(phi.dim() == 4, "Expected 4d tensor")

  const int NZ = u.size(0);
  const int NY = u.size(1);
  const int NX = u.size(2);
  const float LZ = meshInfo.getLZ();
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto u_warped = torch::zeros({NZ,NY,NX}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

   switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warp3d_nearest", ([&]{
          cuda_warp3d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            boundary,
            u_warped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warp3d_trilinear", ([&]{
          cuda_warp3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            boundary,
            u_warped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warp3d_tricubic", ([&]{
          cuda_warp3d_tricubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            boundary,
            u_warped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

  } //end switch interpolation


#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}



std::vector<torch::Tensor> cuda_warp3d_backward( 
  const torch::Tensor u, 
  const torch::Tensor phi, 
  const torch::Tensor forward_out, 
  const MeshInfo3D& meshInfo, 
  const InterpolationType interpolation,
  const BoundaryType boundary)
{

  TORCH_CHECK(u.dim() == 3, "Expected 3d tensor");
  TORCH_CHECK(phi.dim() == 4, "Expected 4d tensor");
  TORCH_CHECK(forward_out.dim() == 3, "Expected 3d tensor")

  const int NZ = u.size(0);
  const int NY = u.size(1);
  const int NX = u.size(2);
  const float LZ = meshInfo.getLZ();
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto u_grad = torch::zeros_like(u);
  auto phi_grad = torch::zeros_like(phi);

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

switch(interpolation)
{
    case INTERPOLATE_NEAREST:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp3d_nearest_backward", ([&]{
            cuda_warp3d_nearest_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              forward_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, 
              boundary,
              u_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              phi_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
              );
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_LINEAR:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp3d_trilinear_backward", ([&]{
            cuda_warp3d_trilinear_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              forward_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, 
              boundary,
              u_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              phi_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
              );
          }));
          cudaSafeCall(cudaGetLastError());
    break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
          AT_DISPATCH_FLOATING_TYPES(u.type(), "warp3d_tricubic_backward", ([&]{
            cuda_warp3d_tricubicHermiteSpline_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              forward_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, 
              boundary,
              u_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
              phi_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
              );
          }));
          cudaSafeCall(cudaGetLastError());
    break;

} //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "backward time " << cut.elapsed() << std::endl;
#endif

  return {u_grad, phi_grad};
}


torch::Tensor cuda_warpVectorField3d(
  const torch::Tensor &u,
  const torch::Tensor &phi,
  const MeshInfo3D& meshInfo,
  const InterpolationType interpolation,
  const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 4, "Expected 4d tensor");
  TORCH_CHECK(phi.dim() == 4, "Expected 4d tensor")

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

  auto u_warped = torch::zeros({NZ,NY,NX,numComp}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warpVectorField3d_nearest", ([&]{
          cuda_warpVectorField3d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            numComp,
            boundary,
            u_warped.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warpVectorField3d_trilinear", ([&]{
          cuda_warpVectorField3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            numComp,
            boundary,
            u_warped.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warpVectorField3d_tricubic", ([&]{
          cuda_warpVectorField3d_tricubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            numComp,
            boundary,
            u_warped.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}

