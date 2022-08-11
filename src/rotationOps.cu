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
__global__ void cuda_rotate3d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rotationMat,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> offset,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_rotate)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T coord_x = ix * hX;
    const T coord_y = iy * hY;
    const T coord_z = iz * hZ;

    const T coord_x_rotated = rotationMat[0][0] * coord_x + rotationMat[0][1] * coord_y + rotationMat[0][2] * coord_z + offset[0];
    const T coord_y_rotated = rotationMat[1][0] * coord_x + rotationMat[1][1] * coord_y + rotationMat[1][2] * coord_z + offset[1];
    const T coord_z_rotated = rotationMat[2][0] * coord_x + rotationMat[2][1] * coord_y + rotationMat[2][2] * coord_z + offset[2];

    u_rotate[iz][iy][ix] = cuda_interpolate3d_nearest(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_rotated, coord_y_rotated, coord_x_rotated);
  }
}

template <typename T>
__global__ void cuda_rotateVectorField3d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rotationMat,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> offset,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_rotate)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T coord_x = ix * hX;
    const T coord_y = iy * hY;
    const T coord_z = iz * hZ;

    const T coord_x_rotated = rotationMat[0][0] * coord_x + rotationMat[0][1] * coord_y + rotationMat[0][2] * coord_z + offset[0];
    const T coord_y_rotated = rotationMat[1][0] * coord_x + rotationMat[1][1] * coord_y + rotationMat[1][2] * coord_z + offset[1];
    const T coord_z_rotated = rotationMat[2][0] * coord_x + rotationMat[2][1] * coord_y + rotationMat[2][2] * coord_z + offset[2];

    for(int comp=0; comp<numComp; ++comp)
    {
      u_rotate[iz][iy][ix][comp] = cuda_interpolateVectorField3d_nearest(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_rotated, coord_y_rotated, coord_x_rotated, comp);
    }
  }
}

template <typename T>
__global__ void cuda_rotateMatrixField3d_nearest_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rotationMat,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> offset,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int numComp_i, const int numComp_j,
  const int boundary,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u_rotate)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T coord_x = ix * hX;
    const T coord_y = iy * hY;
    const T coord_z = iz * hZ;

    const T coord_x_rotated = rotationMat[0][0] * coord_x + rotationMat[0][1] * coord_y + rotationMat[0][2] * coord_z + offset[0];
    const T coord_y_rotated = rotationMat[1][0] * coord_x + rotationMat[1][1] * coord_y + rotationMat[1][2] * coord_z + offset[1];
    const T coord_z_rotated = rotationMat[2][0] * coord_x + rotationMat[2][1] * coord_y + rotationMat[2][2] * coord_z + offset[2];
    
    for(int comp_i=0; comp_i<numComp_i; ++comp_i)
      for(int comp_j=0; comp_j<numComp_j; ++comp_j)
      {
        u_rotate[iz][iy][ix][comp_i][comp_j] = cuda_interpolateMatrixField3d_nearest(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_rotated, coord_y_rotated, coord_x_rotated, comp_i, comp_j);
      }
   }
}

//==========================================
// linear
//==========================================

template <typename T>
__global__ void cuda_rotate3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rotationMat,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> offset,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_rotate)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T coord_x = ix * hX;
    const T coord_y = iy * hY;
    const T coord_z = iz * hZ;

    const T coord_x_rotated = rotationMat[0][0] * coord_x + rotationMat[0][1] * coord_y + rotationMat[0][2] * coord_z + offset[0];
    const T coord_y_rotated = rotationMat[1][0] * coord_x + rotationMat[1][1] * coord_y + rotationMat[1][2] * coord_z + offset[1];
    const T coord_z_rotated = rotationMat[2][0] * coord_x + rotationMat[2][1] * coord_y + rotationMat[2][2] * coord_z + offset[2];

    u_rotate[iz][iy][ix] = cuda_interpolate3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_rotated, coord_y_rotated, coord_x_rotated);
  }
}

template <typename T>
__global__ void cuda_rotateVectorField3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rotationMat,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> offset,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_rotate)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T coord_x = ix * hX;
    const T coord_y = iy * hY;
    const T coord_z = iz * hZ;

    const T coord_x_rotated = rotationMat[0][0] * coord_x + rotationMat[0][1] * coord_y + rotationMat[0][2] * coord_z + offset[0];
    const T coord_y_rotated = rotationMat[1][0] * coord_x + rotationMat[1][1] * coord_y + rotationMat[1][2] * coord_z + offset[1];
    const T coord_z_rotated = rotationMat[2][0] * coord_x + rotationMat[2][1] * coord_y + rotationMat[2][2] * coord_z + offset[2];

    for(int comp=0; comp<numComp; ++comp)
    {
      u_rotate[iz][iy][ix][comp] = cuda_interpolateVectorField3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_rotated, coord_y_rotated, coord_x_rotated, comp);
    }
  }
}


template <typename T>
__global__ void cuda_rotateMatrixField3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rotationMat,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> offset,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int numComp_i, const int numComp_j,
  const int boundary,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u_rotate)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T coord_x = ix * hX;
    const T coord_y = iy * hY;
    const T coord_z = iz * hZ;

    const T coord_x_rotated = rotationMat[0][0] * coord_x + rotationMat[0][1] * coord_y + rotationMat[0][2] * coord_z + offset[0];
    const T coord_y_rotated = rotationMat[1][0] * coord_x + rotationMat[1][1] * coord_y + rotationMat[1][2] * coord_z + offset[1];
    const T coord_z_rotated = rotationMat[2][0] * coord_x + rotationMat[2][1] * coord_y + rotationMat[2][2] * coord_z + offset[2];

    for(int comp_i=0; comp_i<numComp_i; ++comp_i)
      for(int comp_j=0; comp_j<numComp_j; ++comp_j)
      {
       u_rotate[iz][iy][ix][comp_i][comp_j] = cuda_interpolateMatrixField3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_rotated, coord_y_rotated, coord_x_rotated, comp_i, comp_j);
      }
  }
}

//==========================================
// cubic hermite spline
//==========================================


template <typename T>
__global__ void cuda_rotate3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rotationMat,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> offset,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int boundary,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u_rotate)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T coord_x = ix * hX;
    const T coord_y = iy * hY;
    const T coord_z = iz * hZ;

    const T coord_x_rotated = rotationMat[0][0] * coord_x + rotationMat[0][1] * coord_y + rotationMat[0][2] * coord_z + offset[0];
    const T coord_y_rotated = rotationMat[1][0] * coord_x + rotationMat[1][1] * coord_y + rotationMat[1][2] * coord_z + offset[1];
    const T coord_z_rotated = rotationMat[2][0] * coord_x + rotationMat[2][1] * coord_y + rotationMat[2][2] * coord_z + offset[2];

    u_rotate[iz][iy][ix] = cuda_interpolate3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_rotated, coord_y_rotated, coord_x_rotated);
  }
}

template <typename T>
__global__ void cuda_rotateVectorField3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rotationMat,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> offset,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int numComp,
  const int boundary,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u_rotate)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T coord_x = ix * hX;
    const T coord_y = iy * hY;
    const T coord_z = iz * hZ;

    const T coord_x_rotated = rotationMat[0][0] * coord_x + rotationMat[0][1] * coord_y + rotationMat[0][2] * coord_z + offset[0];
    const T coord_y_rotated = rotationMat[1][0] * coord_x + rotationMat[1][1] * coord_y + rotationMat[1][2] * coord_z + offset[1];
    const T coord_z_rotated = rotationMat[2][0] * coord_x + rotationMat[2][1] * coord_y + rotationMat[2][2] * coord_z + offset[2];

    for(int comp=0; comp<numComp; ++comp )
    {
      u_rotate[iz][iy][ix][comp] = cuda_interpolateVectorField3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_rotated, coord_y_rotated, coord_x_rotated, comp);
    }
  }
}


template <typename T>
__global__ void cuda_rotateMatrixField3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rotationMat,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> offset,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
  const int numComp_i, const int numComp_j,
  const int boundary,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u_rotate)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    const T coord_x = ix * hX;
    const T coord_y = iy * hY;
    const T coord_z = iz * hZ;

    const T coord_x_rotated = rotationMat[0][0] * coord_x + rotationMat[0][1] * coord_y + rotationMat[0][2] * coord_z + offset[0];
    const T coord_y_rotated = rotationMat[1][0] * coord_x + rotationMat[1][1] * coord_y + rotationMat[1][2] * coord_z + offset[1];
    const T coord_z_rotated = rotationMat[2][0] * coord_x + rotationMat[2][1] * coord_y + rotationMat[2][2] * coord_z + offset[2];

    for(int comp_i=0; comp_i<numComp_i; ++comp_i)
      for(int comp_j=0; comp_j<numComp_j; ++comp_j)
      {
        u_rotate[iz][iy][ix][comp_i][comp_j] = cuda_interpolateMatrixField3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, boundary, coord_z_rotated, coord_y_rotated, coord_x_rotated, comp_i, comp_j);
      }
  }
}


// ======================================================
// C++ kernel calls
// ======================================================

torch::Tensor cuda_rotate3d(
  const torch::Tensor &u,
  const torch::Tensor &rotationMat,
  const torch::Tensor &offset,
  const MeshInfo3D& meshInfo,
  const InterpolationType interpolation,
  const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 3, "Expected 3d tensor");
  TORCH_CHECK(rotationMat.dim() == 2, "Expected 2d tensor");
  TORCH_CHECK(offset.dim() == 1, "Expected 2d tensor");

  const int NZ = u.size(0);
  const int NY = u.size(1);
  const int NX = u.size(2);
  const float LZ = meshInfo.getLZ();
  const float LY = meshInfo.getLY();
  const float LX = meshInfo.getLX();
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto u_rotate = torch::zeros({NZ,NY,NX}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

   switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "rotate3d_nearest", ([&]{
          cuda_rotate3d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            rotationMat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            boundary,
            u_rotate.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "rotate3d_trilinear", ([&]{
          cuda_rotate3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            rotationMat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            boundary,
            u_rotate.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "rotate3d_tricubic", ([&]{
          cuda_rotate3d_tricubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            rotationMat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            boundary,
            u_rotate.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

  } //end switch interpolation


#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_rotate;
}


torch::Tensor cuda_rotateVectorField3d(
  const torch::Tensor &u,
  const torch::Tensor &rotationMat,
  const torch::Tensor &offset,
  const MeshInfo3D& meshInfo,
  const InterpolationType interpolation,
  const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 4, "Expected 4d tensor");
  TORCH_CHECK(rotationMat.dim() == 2, "Expected 2d tensor");
  TORCH_CHECK(offset.dim() == 1, "Expected 2d tensor");

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

  auto u_rotate = torch::zeros({NZ,NY,NX,numComp}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "rotateVectorField3d_nearest", ([&]{
          cuda_rotateVectorField3d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            rotationMat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            numComp,
            boundary,
            u_rotate.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "rotateVectorField3d_trilinear", ([&]{
          cuda_rotateVectorField3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            rotationMat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            numComp,
            boundary,
            u_rotate.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "rotateVectorField3d_tricubic", ([&]{
          cuda_rotateVectorField3d_tricubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            rotationMat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            numComp,
            boundary,
            u_rotate.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_rotate;
}




torch::Tensor cuda_rotateMatrixField3d(
  const torch::Tensor &u,
  const torch::Tensor &rotationMat,
  const torch::Tensor &offset,
  const MeshInfo3D& meshInfo,
  const InterpolationType interpolation,
  const BoundaryType boundary )
{
  TORCH_CHECK(u.dim() == 4, "Expected 4d tensor");
  TORCH_CHECK(rotationMat.dim() == 2, "Expected 2d tensor");
  TORCH_CHECK(offset.dim() == 1, "Expected 2d tensor");

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

  auto u_rotate = torch::zeros({NZ,NY,NX,numComp_i,numComp_j}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
    case INTERPOLATE_NEAREST:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "rotateMatrixField3d_nearest", ([&]{
          cuda_rotateMatrixField3d_nearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            rotationMat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            numComp_i, numComp_j,
            boundary,
            u_rotate.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_LINEAR:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "rotateMatrixField3d_trilinear", ([&]{
          cuda_rotateMatrixField3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            rotationMat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            numComp_i, numComp_j,
            boundary,
            u_rotate.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

    case INTERPOLATE_CUBIC_HERMITESPLINE:
        AT_DISPATCH_FLOATING_TYPES(u.type(), "rotateMatrixField3d_tricubic", ([&]{
          cuda_rotateMatrixField3d_tricubicHermiteSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
            u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            rotationMat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            NZ, NY, NX,
            LZ, LY, LX,
            hZ, hY, hX,
            numComp_i, numComp_j,
            boundary,
            u_rotate.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
        }));
        cudaSafeCall(cudaGetLastError());
        break;

  } //end switch interpolation

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_rotate;
}

