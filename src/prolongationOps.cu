#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 

#include <iostream>
#include <stdio.h>

#include "coreDefines.h"
#include "interpolation.cu"

// for debugging
// #define CUDA_ERROR_CHECK
// #define CUDA_TIMING

#define cudaSafeCall( err ) __cnnCudaSafeCall( err, __FILE__, __LINE__ )

inline void __cnnCudaSafeCall( cudaError_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
#endif
  return;
}


#ifdef CUDA_TIMING
class CudaTimer
{
public:
  CudaTimer() 
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~CudaTimer() 
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() 
  {
    cudaEventRecord(start_, 0);
  }

  float elapsed() 
  {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float t = 0;
    cudaEventElapsedTime(&t, start_, stop_);
    return t;
  }

private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};
#endif


// CUDA kernels

template <typename T>
__global__ void cuda_prolongate1d_linear_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const int NX, const float LX, const float hX,
  const int NX_Prolong, const float LX_Prolong, const float hX_Prolong,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    u_prolongated[ix] = cuda_interpolate1d_linear(u, NX, LX, hX, coord_x );
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
    u_prolongated[iy][ix] = cuda_interpolate2d_bilinear(u, NY, NX, LY, LX, hY, hX, coord_y, coord_x);
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
    u_prolongated[iz][iy][ix] = cuda_interpolate3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, coord_z, coord_y, coord_x);
  }
}


template <typename T>
__global__ void cuda_prolongate1d_cubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const int NX, const float LX, const float hX,
  const int NX_Prolong, const float LX_Prolong, const float hX_Prolong,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_prolongated)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX_Prolong )
  {
    const T coord_x_prolongated = ix * hX_Prolong;
    const T coord_x = coord_x_prolongated * LX / LX_Prolong;
    u_prolongated[ix] = cuda_interpolate1d_cubicHermiteSpline(u, NX, LX, hX, coord_x);
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
    u_prolongated[iy][ix] = cuda_interpolate2d_bicubicHermiteSpline(u, NY, NX, LY, LX, hY, hX, coord_y, coord_x);
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
    u_prolongated[iz][iy][ix] = cuda_interpolate3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, coord_z, coord_y, coord_x);
  }

}




// ======================================================
// C++ kernel calls
// ======================================================
torch::Tensor cuda_prolongate1d(
  const torch::Tensor &u,
  const MeshInfo1D& meshInfo,
  const MeshInfo1D& meshInfoProlongated,
  const InterpolationType interpolation = INTERPOLATE_LINEAR )
{
  TORCH_CHECK(u.dim() == 1, "Expected 1d tensor");

  const int NX = meshInfo._NX;
  const float LX = meshInfo._LX;
  const float hX = meshInfo._hX;

  const int NX_Prolong = meshInfoProlongated._NX;
  const float LX_Prolong = meshInfoProlongated._LX;
  const float hX_Prolong = meshInfoProlongated._hX;

  auto u_prolongated = torch::zeros({NX_Prolong}, u.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
  case INTERPOLATE_LINEAR: // fallthrough intended
    AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate1d_linear", ([&]{
      cuda_prolongate1d_linear_kernel<scalar_t><<<numBlocks, blockSize>>>(
        u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        NX, LX, hX,
        NX_Prolong, LX_Prolong, hX_Prolong,
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
        u_prolongated.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;

  }


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
  const InterpolationType interpolation = INTERPOLATE_LINEAR )
{
  TORCH_CHECK(u.dim() == 2, "Expected 2d tensor");

  const int NY = u.size(0);
  const int NX = u.size(1);
  const float LY = meshInfo._LY;
  const float LX = meshInfo._LX;
  const float hY = meshInfo._hY;
  const float hX = meshInfo._hX;

  const int NY_Prolong = meshInfoProlongated._NY;
  const int NX_Prolong = meshInfoProlongated._NX;
  const float LY_Prolong = meshInfoProlongated._LY;
  const float LX_Prolong = meshInfoProlongated._LX;
  const float hY_Prolong = meshInfoProlongated._hY;
  const float hX_Prolong = meshInfoProlongated._hX;

  auto u_prolongated = torch::zeros({NY_Prolong,NX_Prolong}, u.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x, (NY_Prolong + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
  case INTERPOLATE_LINEAR: // fallthrough intended
    AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate2d_bilinear", ([&]{
      cuda_prolongate2d_bilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
        u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        NY, NX,
        LY, LX,
        hY, hX,
        NY_Prolong, NX_Prolong,
        LY_Prolong, LX_Prolong,
        hY_Prolong, hX_Prolong,
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
        u_prolongated.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;

  }

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
  const InterpolationType interpolation = INTERPOLATE_LINEAR )
{
  TORCH_CHECK(u.dim() == 3, "Expected 3d tensor");

  const int NZ = u.size(0);
  const int NY = u.size(1);
  const int NX = u.size(2);
  const float LZ = meshInfo._LY;
  const float LY = meshInfo._LY;
  const float LX = meshInfo._LX;
  const float hZ = meshInfo._hY;
  const float hY = meshInfo._hY;
  const float hX = meshInfo._hX;

  const int NZ_Prolong = meshInfoProlongated._NZ;
  const int NY_Prolong = meshInfoProlongated._NY;
  const int NX_Prolong = meshInfoProlongated._NX;
  const float LZ_Prolong = meshInfoProlongated._LZ;
  const float LY_Prolong = meshInfoProlongated._LY;
  const float LX_Prolong = meshInfoProlongated._LX;
  const float hZ_Prolong = meshInfoProlongated._hZ;
  const float hY_Prolong = meshInfoProlongated._hY;
  const float hX_Prolong = meshInfoProlongated._hX;

  auto u_prolongated = torch::zeros({NZ_Prolong,NY_Prolong,NX_Prolong}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX_Prolong + blockSize.x - 1) / blockSize.x, (NY_Prolong + blockSize.y - 1) / blockSize.y, (NZ_Prolong + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
  case INTERPOLATE_LINEAR: // fallthrough intended
    AT_DISPATCH_FLOATING_TYPES(u.type(), "prolongate3d_trilinear", ([&]{
      cuda_prolongate3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
        u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        NZ, NY, NX,
        LZ, LY, LX,
        hZ, hY, hX,
        NZ_Prolong, NY_Prolong, NX_Prolong,
        LZ_Prolong, LY_Prolong, LX_Prolong,
        hZ_Prolong, hY_Prolong, hX_Prolong,
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
        u_prolongated.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_prolongated;
}