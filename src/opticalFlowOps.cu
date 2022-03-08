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
__global__ void cuda_nabla1d_fd_forward_kernel(
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
    cuda_nabla1d_fd_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
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