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
__global__ void cuda_AnisotropicNabla2d_computeTangentVecs_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> NablaInputImage,
  const int NY, const int NX,
  const float alpha, const float beta,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> scalars,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> normals,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> tangentVecs )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
      T normSqr = 0.;
      for( int derivDir=0; derivDir<2; ++derivDir ){
        normSqr += NablaInputImage[iy][ix][derivDir] * NablaInputImage[iy][ix][derivDir];
      }
      const T norm = sqrtf(normSqr);

      if( norm > 0. )
      {
        scalars[iy][ix] = expf( -alpha * powf(norm,beta) );
        for( int derivDir=0; derivDir<2; ++derivDir ){
           normals[iy][ix][derivDir] = NablaInputImage[iy][ix][derivDir] / norm;
        }
        tangentVecs[iy][ix][0]  = -normals[iy][ix][1];
        tangentVecs[iy][ix][1]  = normals[iy][ix][0];
      }else{
        scalars[iy][ix] = 0.;
        for( int derivDir=0; derivDir<2; ++derivDir ){
           normals[iy][ix][derivDir] = 0.;
           tangentVecs[iy][ix][derivDir] = 0.;
        }
      }
  }
}


template <typename T>
__global__ void cuda_AnisotropicNabla2d_forwardVectorField_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Input,
  const int NY, const int NX,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> scalars,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> normals,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> tangentVecs,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Output )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
      for( int flowDir=0; flowDir<2; ++flowDir ){
        T normalFac = 0.;
        T tangentFac = 0.;
        for( int derivDir=0; derivDir<2; ++derivDir ){
          normalFac += normals[iy][ix][derivDir] * Input[iy][ix][flowDir][derivDir];
          tangentFac += tangentVecs[iy][ix][derivDir] * Input[iy][ix][flowDir][derivDir];
        }

        for( int derivDir=0; derivDir<2; ++derivDir ){
          Output[iy][ix][flowDir][derivDir] = scalars[iy][ix] * normalFac * normals[iy][ix][derivDir];
          Output[iy][ix][flowDir][derivDir] += tangentFac * tangentVecs[iy][ix][derivDir];
        }
      }
  }
}



template <typename T>
__global__ void cuda_AnisotropicNabla3d_computeTangentVecs_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> NablaInputImage,
  const int NZ, const int NY, const int NX,
  const float alpha, const float beta,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> scalars,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> normals,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> tangentVecs1,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> tangentVecs2 )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
      T normSqr = 0.;
      for( int derivDir=0; derivDir<3; ++derivDir ){
        normSqr += NablaInputImage[iz][iy][ix][derivDir] * NablaInputImage[iz][iy][ix][derivDir];
      }
      const T norm = sqrtf(normSqr);

      if( norm > 0. )
      {
        scalars[iz][iy][ix] = expf( -alpha * powf(norm,beta) );
        for( int derivDir=0; derivDir<3; ++derivDir ){
           normals[iz][iy][ix][derivDir] = NablaInputImage[iz][iy][ix][derivDir] / norm;
        }
        if( normals[iz][iy][ix][2] > 0.5 ){
          tangentVecs1[iz][iy][ix][0]  = normals[iz][iy][ix][2];
          tangentVecs1[iz][iy][ix][1]  = 0.;
          tangentVecs1[iz][iy][ix][2]  = -normals[iz][iy][ix][0];
        }else{
          tangentVecs1[iz][iy][ix][0]  = normals[iz][iy][ix][1];
          tangentVecs1[iz][iy][ix][1]  = -normals[iz][iy][ix][0];
          tangentVecs1[iz][iy][ix][2]  = 0.;
        }

        // v2 = n x v1
        tangentVecs2[iz][iy][ix][0]  = normals[iz][iy][ix][1] *  tangentVecs1[iz][iy][ix][2] - normals[iz][iy][ix][2] *  tangentVecs1[iz][iy][ix][1];
        tangentVecs2[iz][iy][ix][1]  = normals[iz][iy][ix][2] *  tangentVecs1[iz][iy][ix][0] - normals[iz][iy][ix][0] *  tangentVecs1[iz][iy][ix][2];
        tangentVecs2[iz][iy][ix][2]  = normals[iz][iy][ix][0] *  tangentVecs1[iz][iy][ix][1] - normals[iz][iy][ix][1] *  tangentVecs1[iz][iy][ix][0];

      }else{
        scalars[iz][iy][ix] = 0.;
        for( int derivDir=0; derivDir<3; ++derivDir ){
           normals[iz][iy][ix][derivDir] = 0.;
           tangentVecs1[iz][iy][ix][derivDir] = 0.;
           tangentVecs2[iz][iy][ix][derivDir] = 0.;
        }
      }
  }
}

template <typename T>
__global__ void cuda_AnisotropicNabla3d_forwardVectorField_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> Input,
  const int NZ, const int NY, const int NX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> scalars,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> normals,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> tangentVecs1,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> tangentVecs2,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> Output )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
      for( int flowDir=0; flowDir<3; ++flowDir ){
        T normalFac = 0.;
        T tangent1Fac = 0.;
        T tangent2Fac = 0.;
        for( int derivDir=0; derivDir<3; ++derivDir ){
          normalFac += normals[iz][iy][ix][derivDir] * Input[iz][iy][ix][flowDir][derivDir];
          tangent1Fac += tangentVecs1[iz][iy][ix][derivDir] * Input[iz][iy][ix][flowDir][derivDir];
        }

        for( int derivDir=0; derivDir<3; ++derivDir ){
          Output[iz][iy][ix][flowDir][derivDir] = scalars[iz][iy][ix] * normalFac * normals[iz][iy][ix][derivDir];
          Output[iz][iy][ix][flowDir][derivDir] += tangent1Fac * tangentVecs1[iz][iy][ix][derivDir];
          Output[iz][iy][ix][flowDir][derivDir] += tangent2Fac * tangentVecs2[iz][iy][ix][derivDir];
        }
      }
  }
}






//=========================================================
// C++ kernel calls
//=========================================================



std::vector<torch::Tensor> cuda_AnisotropicNabla2d_computeTangentVecs(
  const torch::Tensor &NablaInputImage, 
  const MeshInfo2D &meshInfo, 
  const float alpha, const float beta )
{
  TORCH_CHECK(NablaInputImage.dim() == 3, "Expected 3d tensor");

  const int NY = NablaInputImage.size(0);
  const int NX = NablaInputImage.size(1);

  auto scalars = torch::zeros({NY,NX}, NablaInputImage.options());
  auto normals = torch::zeros({NY,NX, 2}, NablaInputImage.options());
  auto tangentVecs = torch::zeros({NY,NX, 2}, NablaInputImage.options());

  const dim3 blockSize(32,32,1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y );
#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES( NablaInputImage.type(), "AnisotropicNabla2d", ([&]{
      cuda_AnisotropicNabla2d_computeTangentVecs_kernel<scalar_t><<<numBlocks, blockSize>>>(
        NablaInputImage.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        NY,NX,
        alpha, beta,
        scalars.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        normals.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        tangentVecs.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
        );
  } ) );
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return {scalars,normals,tangentVecs};
}


torch::Tensor cuda_AnisotropicNabla2d_forwardVectorField( 
  const torch::Tensor &Input, 
  const MeshInfo2D &meshInfo, 
  const torch::Tensor &scalars,  const torch::Tensor &normals,  const torch::Tensor &tangents)
{

  TORCH_CHECK(Input.dim() == 4, "Expected 4d tensor");

  const int NY = Input.size(0);
  const int NX = Input.size(1);

  auto output = torch::zeros({NY,NX,2,2}, Input.options());

  const dim3 blockSize(32,32,1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y );
#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES( Input.type(), "AnisotropicNabla2d", ([&]{
      cuda_AnisotropicNabla2d_forwardVectorField_kernel<scalar_t><<<numBlocks, blockSize>>>(
        Input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        NY,NX,
        scalars.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        normals.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        tangents.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
        );
  } ) );
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return output;
}




std::vector<torch::Tensor> cuda_AnisotropicNabla3d_computeTangentVecs(
  const torch::Tensor &NablaInputImage, 
  const MeshInfo3D &meshInfo, 
  const float alpha, const float beta )
{
  TORCH_CHECK(NablaInputImage.dim() == 4, "Expected 4d tensor");

  const int NZ = NablaInputImage.size(0);
  const int NY = NablaInputImage.size(1);
  const int NX = NablaInputImage.size(2);

  auto scalars = torch::zeros({NZ,NY,NX}, NablaInputImage.options());
  auto normals = torch::zeros({NZ,NY,NX, 3}, NablaInputImage.options());
  auto tangentVecs1 = torch::zeros({NZ,NY,NX, 3}, NablaInputImage.options());
  auto tangentVecs2 = torch::zeros({NZ,NY,NX, 3}, NablaInputImage.options());

  const dim3 blockSize(16,16,3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y, 
                       (NZ + blockSize.z - 1) / blockSize.z);
#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES( NablaInputImage.type(), "AnisotropicNabla3d", ([&]{
      cuda_AnisotropicNabla3d_computeTangentVecs_kernel<scalar_t><<<numBlocks, blockSize>>>(
        NablaInputImage.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        NZ,NY,NX,
        alpha, beta,
        scalars.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        normals.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tangentVecs1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tangentVecs2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
        );
  } ) );
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return {scalars,normals,tangentVecs1,tangentVecs2};
}


torch::Tensor cuda_AnisotropicNabla3d_forwardVectorField( 
  const torch::Tensor &Input, 
  const MeshInfo3D &meshInfo, 
  const torch::Tensor &scalars,  const torch::Tensor &normals,  const torch::Tensor &tangents1, const torch::Tensor &tangents2)
{

  TORCH_CHECK(Input.dim() == 5, "Expected 5d tensor");

  const int NZ = Input.size(0);
  const int NY = Input.size(1);
  const int NX = Input.size(2);

  auto output = torch::zeros({NZ,NY,NX,3,3}, Input.options());
  const dim3 blockSize(16,16,3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y, 
                       (NZ + blockSize.z - 1) / blockSize.z);
#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES( Input.type(), "AnisotropicNabla3d", ([&]{
      cuda_AnisotropicNabla3d_forwardVectorField_kernel<scalar_t><<<numBlocks, blockSize>>>(
        Input.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        NZ,NY,NX,
        scalars.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        normals.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tangents1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tangents2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>()
        );
  } ) );
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return output;
}