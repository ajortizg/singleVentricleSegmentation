#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 

#include <iostream>
#include <stdio.h>

#include "coreDefines.h"
#include "interpolation.cuh"
#include "boundary.cuh"
#include "cudaDebug.cuh"



// CUDA kernels

//==========================================
// nearest
//==========================================
template <typename T>
__global__ void cuda_warpCNN3d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> phi,
  const int numBatches, const int numChannels,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    for(int batch=0; batch<numBatches; ++batch)
    {
        const T dx = phi[batch][iz][iy][ix][0];
        const T dy = phi[batch][iz][iy][ix][1];
        const T dz = phi[batch][iz][iy][ix][2];
        const T coord_x_warped = ix * hX + dx;
        const T coord_y_warped = iy * hY + dy;
        const T coord_z_warped = iz * hZ + dz;
        for(int channel=0; channel<numChannels; ++channel)
        {
            u_warped[batch][channel][iz][iy][ix] = cuda_interpolateCNN3d_nearest(u, batch, channel, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped );
        }
    }

    
  }
}

//==========================================
// linear
//==========================================

template <typename T>
__global__ void cuda_warpCNN3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> phi,
  const int numBatches, const int numChannels,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    for(int batch=0; batch<numBatches; ++batch)
    {
        const T dx = phi[batch][iz][iy][ix][0];
        const T dy = phi[batch][iz][iy][ix][1];
        const T dz = phi[batch][iz][iy][ix][2];
        const T coord_x_warped = ix * hX + dx;
        const T coord_y_warped = iy * hY + dy;
        const T coord_z_warped = iz * hZ + dz;
        for(int channel=0; channel<numChannels; ++channel)
        {
            u_warped[batch][channel][iz][iy][ix] = cuda_interpolateCNN3d_trilinear(u, batch, channel, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped );
        }
    }
  }
}

//==========================================
// cubic hermite spline
//==========================================
template <typename T>
__global__ void cuda_warpCNN3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> phi,
  const int numBatches, const int numChannels,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    for(int batch=0; batch<numBatches; ++batch)
    {
        const T dx = phi[batch][iz][iy][ix][0];
        const T dy = phi[batch][iz][iy][ix][1];
        const T dz = phi[batch][iz][iy][ix][2];
        const T coord_x_warped = ix * hX + dx;
        const T coord_y_warped = iy * hY + dy;
        const T coord_z_warped = iz * hZ + dz;
        for(int channel=0; channel<numChannels; ++channel)
        {
            u_warped[batch][channel][iz][iy][ix] = cuda_interpolateCNN3d_tricubicHermiteSpline(u, batch, channel, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_warped, coord_y_warped, coord_x_warped );
        }
    }
  }
}

// ======================================================
// C++ kernel calls
// ======================================================
torch::Tensor cuda_warpCNN3d(
  const torch::Tensor &u,
  const torch::Tensor &phi,
  const MeshInfo3D& meshInfo,
  const InterpolationType interpolation,
  const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 5, "Expected 5d tensor");
  TORCH_CHECK(phi.dim() == 5, "Expected 5d tensor")

  const int numBatches = u.size(0);
  const int numChannels = u.size(1);
  const int NZ = u.size(2);
  const int NY = u.size(3);
  const int NX = u.size(4);
  const float LZ = meshInfo.getLZ();
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto u_warped = torch::zeros({numBatches,numChannels,NZ,NY,NX}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

   switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warpCNN3d_nearest", ([&]{
          cuda_warpCNN3d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            numBatches, numChannels,
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            boundary,
            u_warped.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warpCNN3d_trilinear", ([&]{
          cuda_warpCNN3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            numBatches, numChannels,
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            boundary,
            u_warped.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "warpCNN3d_tricubic", ([&]{
          cuda_warpCNN3d_tricubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            phi.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            numBatches, numChannels,
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            boundary,
            u_warped.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
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