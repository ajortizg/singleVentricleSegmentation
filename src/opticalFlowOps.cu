#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 
#include <iostream>
#include <stdio.h>

#include "coreDefines.h"

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



//=========================================================
// CUDA kernels
//=========================================================

template <typename T>
__global__ void cuda_TVL1OF_threshold_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> rho,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> I1_warped_gradx,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> I1_warped_grady,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> I1_warped_gradz,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  const float LT,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> v)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
    
    const T r = rho[iz][iy][ix];
    const T g2 = I1_warped_gradx[iz][iy][ix]*I1_warped_gradx[iz][iy][ix] + I1_warped_grady[iz][iy][ix]*I1_warped_grady[iz][iy][ix] + I1_warped_gradz[iz][iy][ix]*I1_warped_gradz[iz][iy][ix];

    T delta_x = 0., delta_y = 0., delta_z = 0.;
    if (r < - LT * g2){
        delta_x = LT * I1_warped_gradx[iz][iy][ix];
        delta_y = LT * I1_warped_grady[iz][iy][ix];
        delta_z = LT * I1_warped_gradz[iz][iy][ix];
    } else if(r > LT * g2){
        delta_x = -LT * I1_warped_gradx[iz][iy][ix];
        delta_y = -LT * I1_warped_grady[iz][iy][ix];
        delta_z = -LT * I1_warped_gradz[iz][iy][ix];
    }else if(g2 > 1e-10){
        delta_x = - r * I1_warped_gradx[iz][iy][ix] / g2;
        delta_y = - r * I1_warped_grady[iz][iy][ix] / g2;
        delta_z = - r * I1_warped_gradz[iz][iy][ix] / g2;
    }

    v[iz][iy][ix][0] = u[iz][iy][ix][0] + delta_x;
    v[iz][iy][ix][1] = u[iz][iy][ix][1] + delta_y;
    v[iz][iy][ix][2] = u[iz][iy][ix][2] + delta_z;
  }
}


// template <typename T>
// __global__ void cuda_TVL1OF_dualVariable_kernel(
//   torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
//   const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> v,
//   torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> p,
//   const int NZ, const int NY, const int NX,
//   const float hZ, const float hY, const float hX,
//   const float TAU, const float THETA,
//   const int MAX_INNER_ITERATIONS )
// {
//   int ix = blockDim.x * blockIdx.x + threadIdx.x;
//   int iy = blockDim.y * blockIdx.y + threadIdx.y;
//   int iz = blockDim.z * blockIdx.z + threadIdx.z;

//   if (ix < NX && iy < NY && iz < NZ)
//   {
    
//         nablaOp = opticalFlow.Nabla3D_CD(meshInfo)
//         for m in range(MAX_INNER_ITERATIONS):
//             # Divergence of dual variables
//             p_div_x = nablaOp.backward(p[:,:,:,:,0].contiguous())
//             p_div_y = nablaOp.backward(p[:,:,:,:,1].contiguous())
//             p_div_z = nablaOp.backward(p[:,:,:,:,2].contiguous())
//             print("p_div.norm = ", math.sqrt(torch.norm(p_div_x).item()**2 + torch.norm(p_div_y).item()**2 + torch.norm(p_div_z).item()**2) )

//             # Compute the 3D optical flow Eq. 14 # TODO! check sign
//             u[:,:,:,0] = v[:,:,:,0] - THETA * p_div_x 
//             u[:,:,:,1] = v[:,:,:,1] - THETA * p_div_y
//             u[:,:,:,2] = v[:,:,:,2] - THETA * p_div_z 
//             print("u.norm = ", torch.norm(u).item() )

//             # Proposition 1
//             # Compute the gradient of the optical flow using forward differences
//             # nabla_fwd = NablaForward()
//             u_gradx = nablaOp.forward(u[:, :, :, 0].contiguous())
//             u_grady = nablaOp.forward(u[:, :, :, 1].contiguous())
//             u_gradz = nablaOp.forward(u[:, :, :, 2].contiguous())

//             p_tilde_x = p[:,:,:,:,0] + (TAU / THETA) * u_gradx
//             p_tilde_y = p[:,:,:,:,1] + (TAU / THETA) * u_grady
//             p_tilde_z = p[:,:,:,:,2] + (TAU / THETA) * u_gradz

//             den_x = max(1., p_tilde_x.norm().item())
//             den_y = max(1., p_tilde_y.norm().item())
//             den_z = max(1., p_tilde_z.norm().item())

//             p[:, :, :, :,0] = p_tilde_x / den_x
//             p[:, :, :, :,1] = p_tilde_y / den_y
//             p[:, :, :, :,2] = p_tilde_z / den_z
//   }
// }

template <typename T>
__global__ void cuda_TVL1OF_proxPrimal_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> primalVariable,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> rho,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> I1_warped_grad,
  // const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> I1_warped_grady,
  // const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> I1_warped_gradz,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  const float factor,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> output)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  // if (ix < NX && iy < NY && iz < NZ)
  // {
    
  //   const T r = rho[iz][iy][ix];
  //   const T g2 = I1_warped_gradx[iz][iy][ix]*I1_warped_gradx[iz][iy][ix] + I1_warped_grady[iz][iy][ix]*I1_warped_grady[iz][iy][ix] + I1_warped_gradz[iz][iy][ix]*I1_warped_gradz[iz][iy][ix];

  //   T delta_x = 0., delta_y = 0., delta_z = 0.;
  //   if (r < - factor * g2){
  //       delta_x = factor * I1_warped_gradx[iz][iy][ix];
  //       delta_y = factor * I1_warped_grady[iz][iy][ix];
  //       delta_z = factor * I1_warped_gradz[iz][iy][ix];
  //   } else if(r > factor * g2){
  //       delta_x = -factor * I1_warped_gradx[iz][iy][ix];
  //       delta_y = -factor * I1_warped_grady[iz][iy][ix];
  //       delta_z = -factor * I1_warped_gradz[iz][iy][ix];
  //   }else if(g2 > 1e-10){
  //       delta_x = - r * I1_warped_gradx[iz][iy][ix] / g2;
  //       delta_y = - r * I1_warped_grady[iz][iy][ix] / g2;
  //       delta_z = - r * I1_warped_gradz[iz][iy][ix] / g2;
  //   }

  //   output[iz][iy][ix][0] = primalVariable[iz][iy][ix][0] + delta_x;
  //   output[iz][iy][ix][1] = primalVariable[iz][iy][ix][1] + delta_y;
  //   output[iz][iy][ix][2] = primalVariable[iz][iy][ix][2] + delta_z;
  // }

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
__global__ void cuda_TVL1OF_proxDual_kernel(
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
    
     
    for( int il=0; il<3; ++il )
    {
      T normSqr = 0.;
      for( int ik=0; ik<3; ++ik)
        normSqr += dualVariable[iz][iy][ix][ik][il] * dualVariable[iz][iy][ix][ik][il];
      // const T norm = sqrtf(normSqr);
      // const T den = ( dualFctWeight_TV > norm ) ? dualFctWeight_TV : norm;
      const T den = max(dualFctWeight_TV,  sqrtf(normSqr) );
      for( int ik=0; ik<3; ++ik)
      {
        output[iz][iy][ix][ik][il] = dualVariable[iz][iy][ix][ik][il] / den;
      }
    }

  }
}


//=========================================================
// C++ kernel calls
//=========================================================

torch::Tensor cuda_TVL1OF_threshold( const torch::Tensor &u, const torch::Tensor &rho, 
                                     const torch::Tensor &I1_warped_gradx, const torch::Tensor &I1_warped_grady, const torch::Tensor &I1_warped_gradz,  
                                     const float LT,
                                     const MeshInfo3D &meshInfo)

{
  TORCH_CHECK(u.dim() == 4, "Expected 4 tensor");

  const int NX = meshInfo.getNX();
  const int LX = meshInfo.getLX();
  const float hX = meshInfo.gethX();

  const int NY = meshInfo.getNY();
  const int LY = meshInfo.getLY();
  const float hY = meshInfo.gethY();

  const int NZ = meshInfo.getNZ();
  const int LZ = meshInfo.getLZ();
  const float hZ = meshInfo.gethZ();

  auto v = torch::zeros({NZ,NY,NX,3}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(u.type(), "TVL1OF_threshold", ([&]{
    cuda_TVL1OF_threshold_kernel<scalar_t><<<numBlocks, blockSize>>>(
      u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      rho.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      I1_warped_gradx.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      I1_warped_grady.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      I1_warped_gradz.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      NZ, NY, NX, 
      hZ, hY, hX,
      LT,
      v.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return v;
}





// void cuda_TVL1OF_updateDualVariable(torch::Tensor &u, const torch::Tensor &v, torch::Tensor &p,
//                                  const float TAU, const float THETA
//                                  const MeshInfo3D &meshInfo)

// {
//   TORCH_CHECK(u.dim() == 4, "Expected 4 tensor");

//   const int NX = meshInfo.getNX();
//   const int LX = meshInfo.getLX();
//   const float hX = meshInfo.gethX();

//   const int NY = meshInfo.getNY();
//   const int LY = meshInfo.getLY();
//   const float hY = meshInfo.gethY();

//   const int NZ = meshInfo.getNZ();
//   const int LZ = meshInfo.getLZ();
//   const float hZ = meshInfo.gethZ();

//   const dim3 blockSize(16, 16, 3); 
//   const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

// #ifdef CUDA_TIMING
//   CudaTimer cut;
//   cut.start();
// #endif

//   AT_DISPATCH_FLOATING_TYPES(u.type(), "TVL1OF_updateDualVariable", ([&]{
//     cuda_TVL1OF_updateDualVariable_kernel<scalar_t><<<numBlocks, blockSize>>>(
//       u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//       v.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//       p.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//       NZ, NY, NX, 
//       hZ, hY, hX,
//       TAU, THETA);
//   }));
//   cudaSafeCall(cudaGetLastError());

// #ifdef CUDA_TIMING
//   cudaDeviceSynchronize();
//   std::cout << "forward time " << cut.elapsed() << std::endl;
// #endif

// }



torch::Tensor cuda_TVL1OF_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad,
                                      //  const torch::Tensor &I1_warped_grady, const torch::Tensor &I1_warped_gradz,  
                                      const float weightNorm,
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

  const float factor = primalStepSize_tau * primalFctWeight_Matching * weightNorm;

  auto output = torch::zeros({NZ,NY,NX,3}, primalVariable.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(primalVariable.type(), "TVL1OF_proxPrimal", ([&]{
    cuda_TVL1OF_proxPrimal_kernel<scalar_t><<<numBlocks, blockSize>>>(
      primalVariable.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      rho.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      I1_warped_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      // I1_warped_grady.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      // I1_warped_gradz.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
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


torch::Tensor cuda_TVL1OF_proxDual( const torch::Tensor &dualVariable,
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

  AT_DISPATCH_FLOATING_TYPES(dualVariable.type(), "TVL1OF_proxDual", ([&]{
    cuda_TVL1OF_proxDual_kernel<scalar_t><<<numBlocks, blockSize>>>(
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