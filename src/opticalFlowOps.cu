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

// template <typename T>
// __global__ void cuda_TVL1OF2D_PrimalFct_kernel(
//   const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rho,
//   const int NY, const int NX,
//   const float hY, const float hX,
//   const float factor,
//   float* output)
// {
//   int ix = blockDim.x * blockIdx.x + threadIdx.x;
//   int iy = blockDim.y * blockIdx.y + threadIdx.y;

//   if (ix < NX && iy < NY)
//   {
//     const T r = fabs(rho[iy][ix]);
//     atomicAdd(output, factor * r); 
//   }
// }

template <typename T>
__global__ void cuda_TVL1OF2D_proxPrimal_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> primalVariable,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> rho,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> I1_warped_grad,
  const int NY, const int NX,
  const float hY, const float hX,
  const float factor,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> output)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY)
  {
    const T r = rho[iy][ix];

    T g2 = 0.;
    for(int comp=0; comp<2;++comp)
    {
      g2 += I1_warped_grad[iy][ix][comp]*I1_warped_grad[iy][ix][comp];
    }

    T delta[2];
    if (r < - factor * g2){
      for(int comp=0; comp<2;++comp)
      {
       delta[comp] = factor * I1_warped_grad[iy][ix][comp];
      }
    }else if(r > factor * g2){
      for(int comp=0; comp<2;++comp)
      {
       delta[comp] = -factor * I1_warped_grad[iy][ix][comp];
      }
    }else if(g2 > 1e-10){
      for(int comp=0; comp<2;++comp)
      {
       delta[comp] = -r * I1_warped_grad[iy][ix][comp] / g2;
      }
    }else{
      for(int comp=0; comp<2;++comp)
      {
       delta[comp] = 0.;
      }
    }

    for(int comp=0; comp<2;++comp)
    {
      output[iy][ix][comp] = primalVariable[iy][ix][comp] + delta[comp];
    }
  }
}



template <typename T>
__global__ void cuda_TVL1OF2D_proxDual_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> dualVariable,
  const int NY, const int NX,
  const float hY, const float hX,
  const float dualFctWeight_TV,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> output )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY)
  {
    for( int flowDir=0; flowDir<2; ++flowDir)    
    {
      T normSqr = 0.;
      for( int derivDir=0; derivDir<2; ++derivDir )
      {
        normSqr += dualVariable[iy][ix][flowDir][derivDir] * dualVariable[iy][ix][flowDir][derivDir];
      }
      const T den = fmaxf(1.,  sqrtf(normSqr) / dualFctWeight_TV );
      for( int derivDir=0; derivDir<2; ++derivDir )
      {
        output[iy][ix][flowDir][derivDir] = dualVariable[iy][ix][flowDir][derivDir] / den;
      }
    }
  }
}






template <typename T>
__global__ void cuda_TVL1OF3D_proxPrimal_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> primalVariable,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> rho,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> I1_warped_grad,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  const float factor,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> output)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
    const T r = rho[iz][iy][ix];

    T g2 = 0.;
    for(int comp=0; comp<3;++comp)
    {
      g2 += I1_warped_grad[iz][iy][ix][comp]*I1_warped_grad[iz][iy][ix][comp];
    }

    T delta[3];
    if (r < - factor * g2){
      for(int comp=0; comp<3;++comp)
      {
       delta[comp] = factor * I1_warped_grad[iz][iy][ix][comp];
      }
    }else if(r > factor * g2){
      for(int comp=0; comp<3;++comp)
      {
       delta[comp] = -factor * I1_warped_grad[iz][iy][ix][comp];
      }
    }else if(g2 > 1e-10){
      for(int comp=0; comp<3;++comp)
      {
       delta[comp] = -r * I1_warped_grad[iz][iy][ix][comp] / g2;
      }
    }else{
      for(int comp=0; comp<3;++comp)
      {
       delta[comp] = 0.;
      }
    }

    for(int comp=0; comp<3;++comp)
    {
      output[iz][iy][ix][comp] = primalVariable[iz][iy][ix][comp] + delta[comp];
    }
  }
}



template <typename T>
__global__ void cuda_TVL1OF3D_proxDual_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> dualVariable,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  const float dualFctWeight_TV,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> output )
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
    for( int flowDir=0; flowDir<3; ++flowDir)
    {
      T normSqr = 0.;
      for( int derivDir=0; derivDir<3; ++derivDir )
      {
        normSqr += dualVariable[iz][iy][ix][flowDir][derivDir] * dualVariable[iz][iy][ix][flowDir][derivDir];
      }
      //const T den = max(dualFctWeight_TV,  sqrtf(normSqr) );
      const T den = fmaxf(1.,  sqrtf(normSqr) / dualFctWeight_TV );
      for( int derivDir=0; derivDir<3; ++derivDir )
      {
        output[iz][iy][ix][flowDir][derivDir] = dualVariable[iz][iy][ix][flowDir][derivDir] / den;
      }
    }
  }
}









template <typename T>
__global__ void cuda_TVL1SymOF3D_proxPrimal_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> primalVariable,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> rho_const_l,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> rho_vec_l,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> rho_const_r,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> rho_vec_r,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  const float factor,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> output)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
    const T a0 = factor * rho_const_l[iz][iy][ix];
    const T b0 = factor * rho_const_r[iz][iy][ix];
    T a[3];
    T b[3];
    T u[3];
    T aSqr = 0.;
    T bSqr = 0.;
    T ab = 0.;
    T au = 0.;
    T bu = 0.;
    for(int comp=0; comp<3;++comp){
      a[comp] = factor * rho_vec_l[iz][iy][ix][comp];
      b[comp] = factor * rho_vec_l[iz][iy][ix][comp];
      u[comp] = primalVariable[iz][iy][ix][comp];
      aSqr += a[comp] * a[comp];
      bSqr += b[comp] * b[comp];
      ab += a[comp] * b[comp];
      au += a[comp] * u[comp];
      bu += b[comp] * u[comp];
    }
    T rho_a = a0 + au;
    T rho_b = b0 + bu;

    T delta[3];
    if ( (rho_a - aSqr - ab >= 0) && (rho_b - bSqr - ab >= 0) ){
      for(int comp=0; comp<3;++comp){
       delta[comp] = -a[comp] - b[comp];
      }
    }else if ( (rho_a - aSqr + ab >= 0) && (rho_b + bSqr - ab <= 0) ){
      for(int comp=0; comp<3;++comp){
       delta[comp] = -a[comp] + b[comp];
      }
    }else if ( (rho_a + aSqr - ab <= 0) && (rho_b - bSqr + ab >= 0) ){
      for(int comp=0; comp<3;++comp){
       delta[comp] = a[comp] - b[comp];
      }
    }else if ( (rho_a + aSqr + ab <= 0) && (rho_b + bSqr + ab <= 0) ){
      for(int comp=0; comp<3;++comp){
       delta[comp] = a[comp] + b[comp];
      }
    }else if ( (rho_a - aSqr -ab*(rho_b-ab)/bSqr >= 0) && (fabs(rho_b-ab)/bSqr <= 1) ){
      T fac = (rho_b-ab)/bSqr;
      for(int comp=0; comp<3;++comp){
       delta[comp] = -a[comp] - fac * b[comp];
      }
    }else if ( (rho_a + aSqr -ab*(rho_b+ab)/bSqr <= 0) && (fabs(rho_b+ab)/bSqr <= 1) ){
      T fac = (rho_b+ab)/bSqr;
      for(int comp=0; comp<3;++comp){
       delta[comp] = a[comp] - fac * b[comp];
      }
    }else if ( (rho_b - bSqr -ab*(rho_a-ab)/aSqr >= 0) && (fabs(rho_a-ab)/aSqr <= 1) ){
      T fac = (rho_a-ab)/aSqr;
      for(int comp=0; comp<3;++comp){
       delta[comp] = -b[comp] - fac * a[comp];
      }
    }else if ( (rho_b + bSqr -ab*(rho_a+ab)/aSqr <= 0) && (fabs(rho_a+ab)/aSqr <= 1) ){
      T fac = (rho_a+ab)/aSqr;
      for(int comp=0; comp<3;++comp){
       delta[comp] = b[comp] - fac * a[comp];
      }
    }else{
      //TODO throw exception!
      for(int comp=0; comp<3;++comp)
      {
       delta[comp] = 0.;
      }
    }

    for(int comp=0; comp<3;++comp)
    {
      output[iz][iy][ix][comp] = primalVariable[iz][iy][ix][comp] + delta[comp];
    }
  }
}





//=========================================================
// C++ kernel calls
//=========================================================


torch::Tensor cuda_TVL1OF2D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad,
                                      const MeshInfo2D &meshInfo)
{
  TORCH_CHECK(primalVariable.dim() == 3, "Expected 3 tensor");

  const int NX = meshInfo.getNX();
  const int LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  const int NY = meshInfo.getNY();
  const int LY = meshInfo.getLY();
  const float hY = meshInfo.gethY();

  const float factor = primalStepSize_tau * primalFctWeight_Matching;

  auto output = torch::zeros({NY,NX,2}, primalVariable.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(primalVariable.type(), "TVL1OF2D_proxPrimal", ([&]{
    cuda_TVL1OF2D_proxPrimal_kernel<scalar_t><<<numBlocks, blockSize>>>(
      primalVariable.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      rho.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      I1_warped_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      NY, NX, 
      hY, hX,
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


torch::Tensor cuda_TVL1OF2D_proxDual( const torch::Tensor &dualVariable,
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo2D &meshInfo ) 
{

  TORCH_CHECK(dualVariable.dim() == 4, "Expected 4 tensor");

  const int NX = meshInfo.getNX();
  const int LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  const int NY = meshInfo.getNY();
  const int LY = meshInfo.getLY();
  const float hY = meshInfo.gethY();

  auto output = torch::zeros({NY,NX,2,2}, dualVariable.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(dualVariable.type(), "TVL1OF2D_proxDual", ([&]{
    cuda_TVL1OF2D_proxDual_kernel<scalar_t><<<numBlocks, blockSize>>>(
      dualVariable.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      NY, NX, 
      hY, hX,
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








torch::Tensor cuda_TVL1OF3D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad,
                                      const MeshInfo3D &meshInfo)
{
  TORCH_CHECK(primalVariable.dim() == 4, "Expected 4 tensor");

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

  auto output = torch::zeros({NZ,NY,NX,3}, primalVariable.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(primalVariable.type(), "TVL1OF3D_proxPrimal", ([&]{
    cuda_TVL1OF3D_proxPrimal_kernel<scalar_t><<<numBlocks, blockSize>>>(
      primalVariable.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      rho.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      I1_warped_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      NZ, NY, NX, 
      hZ, hY, hX,
      factor,
      output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return output;
}


torch::Tensor cuda_TVL1OF3D_proxDual( const torch::Tensor &dualVariable,
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo3D &meshInfo ) 
{

  TORCH_CHECK(dualVariable.dim() == 5, "Expected 5 tensor");

  const int NX = meshInfo.getNX();
  const int LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  const int NY = meshInfo.getNY();
  const int LY = meshInfo.getLY();
  const float hY = meshInfo.gethY();

  const int NZ = meshInfo.getNZ();
  const int LZ = meshInfo.getLZ();
  const float hZ = meshInfo.gethZ();

  auto output = torch::zeros({NZ,NY,NX,3,3}, dualVariable.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(dualVariable.type(), "TVL1OF3D_proxDual", ([&]{
    cuda_TVL1OF3D_proxDual_kernel<scalar_t><<<numBlocks, blockSize>>>(
      dualVariable.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      NZ, NY, NX, 
      hZ, hY, hX,
      dualFctWeight_TV,
      output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return output;                             
};





torch::Tensor cuda_TVL1SymOF3D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho_const_l, const torch::Tensor &rho_vec_l,
                                      const torch::Tensor &rho_const_r, const torch::Tensor &rho_vec_r,
                                      const MeshInfo3D &meshInfo)
{
  TORCH_CHECK(primalVariable.dim() == 4, "Expected 4 tensor");

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

  auto output = torch::zeros({NZ,NY,NX,3}, primalVariable.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(primalVariable.type(), "TVL1SymOF3D_proxPrimal", ([&]{
    cuda_TVL1SymOF3D_proxPrimal_kernel<scalar_t><<<numBlocks, blockSize>>>(
      primalVariable.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      rho_const_l.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      rho_vec_l.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      rho_const_r.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      rho_vec_r.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      NZ, NY, NX, 
      hZ, hY, hX,
      factor,
      output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return output;
}