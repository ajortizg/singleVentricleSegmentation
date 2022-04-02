#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 
#include <iostream>
#include <stdio.h>

#include "coreDefines.h"
#include "cudaDebug.cuh"


//=========================================================
// CUDA kernels
//=========================================================

template <typename T>
__global__ void cuda_ROF2D_proxPrimal_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> primalVariable,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> inputImage,
  const int NY, const int NX,
  const float hY, const float hX,
  const float factor,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> output)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY)
  {
    output[iy][ix] = (primalVariable[iy][ix] + factor * inputImage[iy][ix])/(1.+factor);
  }
}



template <typename T>
__global__ void cuda_ROF2D_proxDual_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> dualVariable,
  const int NY, const int NX,
  const float hY, const float hX,
  const float dualFctWeight_TV,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> output )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY)
  {
      T normSqr = 0.;
      for( int derivDir=0; derivDir<2; ++derivDir )
      {
        normSqr += dualVariable[iy][ix][derivDir] * dualVariable[iy][ix][derivDir];
      }
      //const T den = max(dualFctWeight_TV,  sqrtf(normSqr) );
      const T den = fmaxf(1.,  sqrtf(normSqr) / dualFctWeight_TV );
      for( int derivDir=0; derivDir<2; ++derivDir )
      {
        output[iy][ix][derivDir] = dualVariable[iy][ix][derivDir] / den;
      }
  }
}






template <typename T>
__global__ void cuda_ROF3D_proxPrimal_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> primalVariable,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> inputImage,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  const float factor,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> output)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
      output[iz][iy][ix] =(primalVariable[iz][iy][ix] + factor * inputImage[iz][iy][ix])/(1.+factor);
  }
}


template <typename T>
__global__ void cuda_ROF3D_proxDual_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> dualVariable,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  const float dualFctWeight_TV,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> output )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
      T normSqr = 0.;
      for( int derivDir=0; derivDir<3; ++derivDir )
      {
        normSqr += dualVariable[iz][iy][ix][derivDir] * dualVariable[iz][iy][ix][derivDir];
      }
      //const T den = max(dualFctWeight_TV,  sqrtf(normSqr) );
      const T den = fmaxf(1.,  sqrtf(normSqr) / dualFctWeight_TV );
      for( int derivDir=0; derivDir<3; ++derivDir )
      {
        output[iz][iy][ix][derivDir] = dualVariable[iz][iy][ix][derivDir] / den;
      }
  }
}


//=========================================================
// C++ kernel calls
//=========================================================


torch::Tensor cuda_ROF2D_proxPrimal( const torch::Tensor &primalVariable,
                                     const torch::Tensor &inputImage,
                                     const float primalStepSize_tau, 
                                     const float primalFctWeight_Matching,
                                     const MeshInfo2D &meshInfo)
{
  TORCH_CHECK(primalVariable.dim() == 2, "Expected 2 tensor");
  TORCH_CHECK(inputImage.dim() == 2, "Expected 2 tensor");

  const int NX = meshInfo.getNX();
  const int LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  const int NY = meshInfo.getNY();
  const int LY = meshInfo.getLY();
  const float hY = meshInfo.gethY();

  const float factor = primalStepSize_tau * primalFctWeight_Matching;

  auto output = torch::zeros({NY,NX}, primalVariable.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(primalVariable.type(), "ROF2D_proxPrimal", ([&]{
    cuda_ROF2D_proxPrimal_kernel<scalar_t><<<numBlocks, blockSize>>>(
      primalVariable.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      inputImage.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      NY, NX, 
      hY, hX,
      factor,
      output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return output;
}


torch::Tensor cuda_ROF2D_proxDual( const torch::Tensor &dualVariable,
                                   const float dualStepSize_sigma,
                                   const float dualFctWeight_TV,
                                   const MeshInfo2D &meshInfo ) 
{

  TORCH_CHECK(dualVariable.dim() == 3, "Expected 3 tensor");

  const int NX = meshInfo.getNX();
  const int LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  const int NY = meshInfo.getNY();
  const int LY = meshInfo.getLY();
  const float hY = meshInfo.gethY();

  auto output = torch::zeros({NY,NX,2}, dualVariable.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(dualVariable.type(), "ROF2D_proxDual", ([&]{
    cuda_ROF2D_proxDual_kernel<scalar_t><<<numBlocks, blockSize>>>(
      dualVariable.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      NY, NX, 
      hY, hX,
      dualFctWeight_TV,
      output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return output;                             
};





torch::Tensor cuda_ROF3D_proxPrimal( const torch::Tensor &primalVariable,
                                     const torch::Tensor &inputImage,
                                     const float primalStepSize_tau, 
                                     const float primalFctWeight_Matching,
                                     const MeshInfo3D &meshInfo)
{
  TORCH_CHECK(primalVariable.dim() == 3, "Expected 3 tensor");

  const int NX = meshInfo.getNX();
  const int LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  const int NY = meshInfo.getNY();
  const int LY = meshInfo.getLY();
  const float hY = meshInfo.gethY();

  const int NZ = meshInfo.getNZ();
  const int LZ = meshInfo.getLZ();
  const float hZ = meshInfo.gethZ();

  const float factor = primalStepSize_tau * primalFctWeight_Matching;

  auto output = torch::zeros({NZ,NY,NX}, primalVariable.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(primalVariable.type(), "ROF3D_proxPrimal", ([&]{
    cuda_ROF3D_proxPrimal_kernel<scalar_t><<<numBlocks, blockSize>>>(
      primalVariable.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      inputImage.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      NZ, NY, NX, 
      hZ, hY, hX,
      factor,
      output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return output;
}


torch::Tensor cuda_ROF3D_proxDual( const torch::Tensor &dualVariable,
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo3D &meshInfo ) 
{

  TORCH_CHECK(dualVariable.dim() == 4, "Expected 4 tensor");

  const int NX = meshInfo.getNX();
  const int LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  const int NY = meshInfo.getNY();
  const int LY = meshInfo.getLY();
  const float hY = meshInfo.gethY();

  const int NZ = meshInfo.getNZ();
  const int LZ = meshInfo.getLZ();
  const float hZ = meshInfo.gethZ();

  auto output = torch::zeros({NZ,NY,NX,3}, dualVariable.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(dualVariable.type(), "ROF3D_proxDual", ([&]{
    cuda_ROF3D_proxDual_kernel<scalar_t><<<numBlocks, blockSize>>>(
      dualVariable.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      NZ, NY, NX, 
      hZ, hY, hX,
      dualFctWeight_TV,
      output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return output;                             
};