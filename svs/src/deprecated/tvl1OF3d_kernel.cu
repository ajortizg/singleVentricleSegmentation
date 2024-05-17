#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 

#include <iostream>


#include "stdio.h"

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


// helpers
__forceinline__ __device__ float myabs(const float x)
{
  return fabsf(x);
}

__forceinline__ __device__ double myabs(const double x)
{
  return fabs(x);
}

// CUDA kernels
// template <typename T>
// __global__ void cuda_primal_update_step_kernel(
//   torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
//   torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
//   torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> ATq,
//   const float tau, //torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> tau, //
//   const float hz,
//   const int Z,
//   const int M,
//   const int N,
//   const int D) 
// {
//   int ix = blockDim.x * blockIdx.x + threadIdx.x;
//   int iy = blockDim.y * blockIdx.y + threadIdx.y;
//   int iz = blockDim.z * blockIdx.z + threadIdx.z;

//   T temp = 0;
//   if (ix < N && iy < M && iz < Z)
//   {
//     // finite differences backward
//     temp += (ix > 0) ? (ix < N - 1) ? p[iz][iy][ix-1][0] - p[iz][iy][ix][0] : p[iz][iy][ix-1][0] : -p[iz][iy][ix][0];
//     temp += (iy > 0) ? (iy < M - 1) ? p[iz][iy-1][ix][1] - p[iz][iy][ix][1] : p[iz][iy-1][ix][1] : -p[iz][iy][ix][1]; 
//     temp += (iz > 0) ? (iz < Z - 1) ? (p[iz-1][iy][ix][2] - p[iz][iy][ix][2])/hz : p[iz-1][iy][ix][2]/hz : -p[iz][iy][ix][2]/hz;

//     u[iz][iy][ix] -= tau*(temp + ATq[iz][iy][ix]);     
//     //u[iz][iy][ix] -= tau[iz][iy][ix]*(temp + ATq[iz][iy][ix]);     
//   }
// }

// template <typename T>
// __global__ void cuda_dual_update_step_kernel(
//   torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
//   torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
//   torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> sigma,
//   const float hz,
//   const float lamda,
//   const int Z,
//   const int M,
//   const int N,
//   const int D) 
// {
//   int ix = blockDim.x * blockIdx.x + threadIdx.x;
//   int iy = blockDim.y * blockIdx.y + threadIdx.y;
//   int iz = blockDim.z * blockIdx.z + threadIdx.z;

//   if (ix < N && iy < M && iz < Z)
//   {
//       const int xp = ix + (ix < N - 1);
//       const int yp = iy + (iy < M - 1);
//       const int zp = iz + (iz < Z - 1);

//       const T ptmp_x = p[iz][iy][ix][0] + sigma[0]*(u[iz][iy][xp] - u[iz][iy][ix]);
//       const T ptmp_y = p[iz][iy][ix][1] + sigma[1]*(u[iz][yp][ix] - u[iz][iy][ix]);
//       const T ptmp_z = p[iz][iy][ix][2] + sigma[2]*(u[zp][iy][ix] - u[iz][iy][ix])/hz;

//       // calculate 2-norm along first dimension
//       const T denom_temp = max(sqrtf(ptmp_x*ptmp_x + ptmp_y*ptmp_y + ptmp_z*ptmp_z)/lamda, 1.);

//       // prox operation
//       p[iz][iy][ix][0] = ptmp_x / denom_temp;
//       p[iz][iy][ix][1] = ptmp_y / denom_temp;
//       p[iz][iy][ix][2] = ptmp_z / denom_temp;
//   }
// }

// template <typename T>
// __global__ void cuda_prox_l2_forward_kernel(
//   torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> q,
//   torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> sigma,
//   const int Z,
//   const int M,
//   const int N) 
// {
//   int ix = blockDim.x * blockIdx.x + threadIdx.x;
//   int iy = blockDim.y * blockIdx.y + threadIdx.y;
//   int iz = blockDim.z * blockIdx.z + threadIdx.z;

//   if (ix < N && iy < M && iz < Z)
//   {
//     // l2 prox on q 
//     q[iz][iy][ix] /= (sigma[iz] + 1.);
//   }
// }




template <typename T>
__global__ void cuda_nabla1d_fd_forward_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> b,
  const int NX,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
      Db[ix][0] = (ix < NX-1) ? (b[ix+1] - b[ix]) : 0;
  }
  
}


template <typename T>
__global__ void cuda_divergence1d_fd_backward_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> p,
  const int NX,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
      divp[ix] = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               (p[ix-1][0] - p[ix][0]) 
                               : 
                               p[ix-1][0]
                              :
                              -p[ix][0];
  }
  
}


template <typename T>
__global__ void cuda_nabla2d_fd_forward_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> b,
  const int NY,
  const int NX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
      Db[iy][ix][0] = (ix < NX-1) ? (b[iy][ix+1] - b[iy][ix]) : 0;
      Db[iy][ix][1] = (iy < NY-1) ? (b[iy+1][ix] - b[iy][ix]) : 0;
  }
  
}


template <typename T>
__global__ void cuda_divergence2d_fd_backward_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> p,
  const int NY,
  const int NX,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY)
  {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               (p[iy][ix-1][0] - p[iy][ix][0]) 
                               : 
                               p[iy][ix-1][0]
                              :
                              -p[iy][ix][0];

      T divp_y = (iy > 0) ? 
                            (iy < NY - 1 ) ? 
                               (p[iy-1][ix][1] - p[iy][ix][1]) 
                               : 
                               p[iy-1][ix][1]
                              :
                              -p[iy][ix][1];
                        
      divp[iy][ix] = divp_x + divp_y;
  }
  
}


template <typename T>
__global__ void cuda_nabla3d_fd_forward_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> b,
  const int NZ,
  const int NY,
  const int NX,
  const float hZ,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
      Db[iz][iy][ix][0] = (ix < NX-1) ? (b[iz][iy][ix+1] - b[iz][iy][ix]) : 0;
      Db[iz][iy][ix][1] = (iy < NY-1) ? (b[iz][iy+1][ix] - b[iz][iy][ix]) : 0;
      Db[iz][iy][ix][2] = (iz < NZ-1) ? (b[iz+1][iy][ix] - b[iz][iy][ix])/hZ : 0;
  }
  
}


template <typename T>
__global__ void cuda_divergence3d_fd_backward_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
  const int NZ,
  const int NY,
  const int NX,
  const float hZ,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               (p[iz][iy][ix-1][0] - p[iz][iy][ix][0]) 
                               : 
                               p[iz][iy][ix-1][0]
                              :
                              -p[iz][iy][ix][0];

      T divp_y = (iy > 0) ? 
                            (iy < NY - 1 ) ? 
                               (p[iz][iy-1][ix][1] - p[iz][iy][ix][1]) 
                               : 
                               p[iz][iy-1][ix][1]
                              :
                              -p[iz][iy][ix][1];
                        
      T divp_z = (iz > 0) ? 
                            (iz < NZ - 1 ) ? 
                               (p[iz-1][iy][ix][2] - p[iz][iy][ix][2]) 
                               : 
                               p[iz-1][iy][ix][2]
                              :
                              -p[iz][iy][ix][2];
                        
      divp[iz][iy][ix] = divp_x + divp_y + divp_z/hZ;
  }
  
}



template <typename T>
__global__ void cuda_nabla1d_cd_forward_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> b,
  const int NX,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
      Db[ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[ix+1] - b[ix-1]) 
                              : 
                              0.5*(b[ix] - b[ix-1]) 
                            : 
                            0.5*(b[ix+1]-b[ix]);
  }
  
}

template <typename T>
__global__ void cuda_divergence1d_cd_backward_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> p,
  const int NX,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
      divp[ix] = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               0.5*(p[ix-1][0] - p[ix+1][0]) 
                               : 
                               0.5*(p[ix-1][0] + p[ix][0]) 
                              :
                              0.5*(-p[ix][0] - p[ix+1][0]);
  }
  
}

template <typename T>
__global__ void cuda_nabla2d_cd_forward_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> b,
  const int NY,
  const int NX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
      Db[iy][ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iy][ix+1] - b[iy][ix-1]) 
                              : 
                              0.5*(b[iy][ix] - b[iy][ix-1]) 
                            : 
                            0.5*(b[iy][ix+1]-b[iy][ix]);

      Db[iy][ix][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iy+1][ix] - b[iy-1][ix]) 
                              : 
                              0.5*(b[iy][ix] - b[iy-1][ix]) 
                            : 
                            0.5*(b[iy+1][ix]-b[iy][ix]);
  }
  
}


template <typename T>
__global__ void cuda_divergence2d_cd_backward_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> p,
  const int NY,
  const int NX,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               0.5*(p[iy][ix-1][0] - p[iy][ix+1][0]) 
                               : 
                               0.5*(p[iy][ix-1][0] + p[iy][ix][0]) 
                              :
                              0.5*(-p[iy][ix][0] - p[iy][ix+1][0]);

      T divp_y = (iy > 0) ? 
                            (iy < NY - 1 ) ? 
                               0.5*(p[iy-1][ix][1] - p[iy+1][ix][1]) 
                               : 
                               0.5*(p[iy-1][ix][1] + p[iy][ix][1]) 
                              :
                              0.5*(-p[iy][ix][1] - p[iy+1][ix][1]);

      divp[iy][ix] = divp_x + divp_y;
  }
  
}


template <typename T>
__global__ void cuda_nabla3d_cd_forward_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> b,
  const int NZ,
  const int NY,
  const int NX,
  const float hZ,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
      Db[iz][iy][ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iz][iy][ix+1] - b[iz][iy][ix-1]) 
                              : 
                              0.5*(b[iz][iy][ix] - b[iz][iy][ix-1]) 
                            : 
                            0.5*(b[iz][iy][ix+1]-b[iz][iy][ix]);

      Db[iz][iy][ix][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iz][iy+1][ix] - b[iz][iy-1][ix]) 
                              : 
                              0.5*(b[iz][iy][ix] - b[iz][iy-1][ix]) 
                            : 
                            0.5*(b[iz][iy+1][ix]-b[iz][iy][ix]);

      Db[iz][iy][ix][2] = (iz > 0) ? 
                            (iz < NZ-1) ? 
                              0.5*(b[iz+1][iy][ix] - b[iz-1][iy][ix])/hZ
                              : 
                              0.5*(b[iz][iy][ix] - b[iz-1][iy][ix])/hZ
                            : 
                            0.5*(b[iz+1][iy][ix]-b[iz][iy][ix])/hZ;
  }
  
}


template <typename T>
__global__ void cuda_divergence3d_cd_backward_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
  const int NZ,
  const int NY,
  const int NX,
  const float hZ,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               0.5*(p[iz][iy][ix-1][0] - p[iz][iy][ix+1][0]) 
                               : 
                               0.5*(p[iz][iy][ix-1][0] + p[iz][iy][ix][0]) 
                              :
                              0.5*(-p[iz][iy][ix][0] - p[iz][iy][ix+1][0]);

      T divp_y = (iy > 0) ? 
                            (iy < NY - 1 ) ? 
                               0.5*(p[iz][iy-1][ix][1] - p[iz][iy+1][ix][1]) 
                               : 
                               0.5*(p[iz][iy-1][ix][1] + p[iz][iy][ix][1]) 
                              :
                              0.5*(-p[iz][iy][ix][1] - p[iz][iy+1][ix][1]);

      T divp_z = (iz > 0) ? 
                            (iz < NZ - 1 ) ? 
                               0.5*(p[iz-1][iy][ix][2] - p[iz+1][iy][ix][2]) 
                               : 
                               0.5*(p[iz-1][iy][ix][2] + p[iz][iy][ix][2]) 
                              :
                              0.5*(-p[iz][iy][ix][2] - p[iz+1][iy][ix][2]);

      divp[iz][iy][ix] = divp_x + divp_y + divp_z/hZ;
  }
  
}



// template <typename T>
// __global__ void alternative_cuda_central_differences_kernel_3D(
//   const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> b,
//   const int NX,
//   const int NY,
//   const int NZ,
//   const float hZ,
//   torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Db)
// {
//   int ix = blockDim.x * blockIdx.x + threadIdx.x;
//   int iy = blockDim.y * blockIdx.y + threadIdx.y;
//   int iz = blockDim.z * blockIdx.z + threadIdx.z;

//   //printf("%d \n", ix);

//   if (ix < NX && iy < NY && iz < NZ)
//   {
//       Db[ix][iy][iz][0] = (ix > 0) ? 
//                             (ix < NX-1) ? 
//                               0.5*(b[ix+1][iy][iz] - b[ix-1][iy][iz]) 
//                               : 
//                               0.5*(b[ix][iy][iz] - b[ix-1][iy][iz]) 
//                             : 
//                             0.5*(b[ix+1][iy][iz]-b[ix][iy][iz]);

//       Db[ix][iy][iz][1] = (iy > 0) ? 
//                             (iy < NY-1) ? 
//                               0.5*(b[ix][iy+1][iz] - b[ix][iy-1][iz]) 
//                               : 
//                               0.5*(b[ix][iy][iz] - b[ix][iy-1][iz]) 
//                             : 
//                             0.5*(b[ix][iy+1][iz]-b[ix][iy][iz]);

//       Db[ix][iy][iz][2] = (iz > 0) ? 
//                             (iz < NZ-1) ? 
//                               0.5*(b[ix][iy][iz+1] - b[ix][iy][iz-1])/hZ
//                               : 
//                               0.5*(b[ix][iy][iz] - b[ix][iy][iz-1])/hZ
//                             : 
//                             0.5*(b[ix][iy][iz+1]-b[ix][iy][iz])/hZ;
//   }
  
// }


// C++ kernel calls
// void cuda_primal_update_step(
//   const torch::Tensor &u,
//   const torch::Tensor &p,
//   const torch::Tensor &ATq,
//   const float tau, //const torch::Tensor &tau, //
//   const float hz)
// {
//   TORCH_CHECK(u.dim() == 3, "Expected 3d tensor for dual p");
//   TORCH_CHECK(p.dim() == 4, "Expected 4d tensor for primal u");
//   TORCH_CHECK(ATq.dim() == 3, "Expected 3d tensor for primal u");
//   //TORCH_CHECK(tau.dim() == 3, "Expected 3d tensor for stepsize tau");

//   const int Z = p.size(0);
//   const int M = p.size(1);
//   const int N = p.size(2);
//   const int D = p.size(3);
  
//   cudaDeviceSynchronize();

//   const dim3 blockSize(32, 32, 1); 
//   const dim3 numBlocks((N + blockSize.x - 1) / blockSize.x,
//                       (M + blockSize.y - 1) / blockSize.y,
//                       (Z + blockSize.z - 1) / blockSize.z);

// #ifdef CUDA_TIMING
//   CudaTimer cut;
//   cut.start();
// #endif

//   AT_DISPATCH_FLOATING_TYPES(p.type(), "primal_update_step", ([&]{
//     cuda_primal_update_step_kernel<scalar_t><<<numBlocks, blockSize>>>(
//       u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//       p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//       ATq.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//       tau, //tau.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), //
//       hz,
//       Z,
//       M,
//       N,
//       D); 
//   }));
//   cudaSafeCall(cudaGetLastError());

// #ifdef CUDA_TIMING
//   cudaDeviceSynchronize();
//   std::cout << "forward time " << cut.elapsed() << std::endl;
// #endif
// }

// void cuda_dual_update_step(
//   const torch::Tensor &p,
//   const torch::Tensor &u,
//   const torch::Tensor &sigma,
//   const float hz,
//   const float lamda)
// {
//   TORCH_CHECK(p.dim() == 4, "Expected 4d tensor for dual p");
//   TORCH_CHECK(u.dim() == 3, "Expected 3d tensor for primal u");
//   TORCH_CHECK(sigma.dim() == 1, "Expected 1d tensor for stepsize sigma");

//   const int Z = p.size(0);
//   const int M = p.size(1);
//   const int N = p.size(2);
//   const int D = p.size(3);

//   const dim3 blockSize(32, 32, 1); 
//   const dim3 numBlocks((N + blockSize.x - 1) / blockSize.x,
//                       (M + blockSize.y - 1) / blockSize.y,
//                       (Z + blockSize.z - 1) / blockSize.z);

// #ifdef CUDA_TIMING
//   CudaTimer cut;
//   cut.start();
// #endif

//   AT_DISPATCH_FLOATING_TYPES(p.type(), "dual_update_step", ([&]{
//     cuda_dual_update_step_kernel<scalar_t><<<numBlocks, blockSize>>>(
//       p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//       u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//       sigma.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
//       hz,
//       lamda,
//       Z,
//       M,
//       N,
//       D); 
//   }));
//   cudaSafeCall(cudaGetLastError());

// #ifdef CUDA_TIMING
//   cudaDeviceSynchronize();
//   std::cout << "forward time " << cut.elapsed() << std::endl;
// #endif
// }

// void cuda_prox_l2(
//   const torch::Tensor &q,
//   const torch::Tensor &sigma)
// {
//   TORCH_CHECK(q.dim() == 3, "Expected 3d tensor");
//   TORCH_CHECK(sigma.dim() == 1, "Expected 1d tensor");

//   const int Z = q.size(0);
//   const int M = q.size(1);
//   const int N = q.size(2);

//   const dim3 blockSize(32, 32, 1); 
//   const dim3 numBlocks((N + blockSize.x - 1) / blockSize.x,
//                       (M + blockSize.y - 1) / blockSize.y,
//                       (Z + blockSize.z - 1) / blockSize.z);

// #ifdef CUDA_TIMING
//   CudaTimer cut;
//   cut.start();
// #endif

//   AT_DISPATCH_FLOATING_TYPES(q.type(), "prox_l2_forward", ([&]{
//     cuda_prox_l2_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
//       q.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//       sigma.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
//       Z,
//       M,
//       N); 
//   }));
//   cudaSafeCall(cudaGetLastError());
  
// #ifdef CUDA_TIMING
//   cudaDeviceSynchronize();
//   std::cout << "forward time " << cut.elapsed() << std::endl;
// #endif
// }








torch::Tensor cuda_nabla1d_fd_forward(
  const torch::Tensor &b)
{
  TORCH_CHECK(b.dim() == 1, "Expected 1d tensor");

  const int NX = b.size(0);

  auto Db = torch::zeros({NX, 1}, b.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla1d_fd_forward", ([&]{
    cuda_nabla1d_fd_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      NX,
      Db.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}


torch::Tensor cuda_divergence1d_fd_backward(
  const torch::Tensor &p)
{
  TORCH_CHECK(p.dim() == 2, "Expected 2d tensor");

  const int NX = p.size(0);

  auto divp = torch::zeros({NX}, p.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence1d_fd_backward", ([&]{
    cuda_divergence1d_fd_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      p.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      NX,
      divp.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}


torch::Tensor cuda_nabla2d_fd_forward(
  const torch::Tensor &b)
{
  TORCH_CHECK(b.dim() == 2, "Expected 2d tensor");

  const int NY = b.size(0);
  const int NX = b.size(1);

  auto Db = torch::zeros({NY, NX, 2}, b.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                      (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla2d_fd_forward", ([&]{
    cuda_nabla2d_fd_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      NY,
      NX,
      Db.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}


torch::Tensor cuda_divergence2d_fd_backward(
  const torch::Tensor &p)
{
  TORCH_CHECK(p.dim() == 3, "Expected 3d tensor");

  const int NY = p.size(0);
  const int NX = p.size(1);

  auto divp = torch::zeros({NY,NX}, p.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y  );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence2d_fd_backward", ([&]{
    cuda_divergence2d_fd_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      p.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      NY,
      NX,
      divp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}

torch::Tensor cuda_nabla3d_fd_forward(
  const torch::Tensor &b)
{
  TORCH_CHECK(b.dim() == 3, "Expected 3d tensor");

  const int NZ = b.size(0);
  const int NY = b.size(1);
  const int NX = b.size(2);
  const float hZ = NZ / std::sqrt(0.5*NX*NX+0.5*NY*NY);

  auto Db = torch::zeros({NZ,NY,NX, 3}, b.options());

  const dim3 blockSize(16,16,3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                      (NY + blockSize.y - 1) / blockSize.y,
                      (NZ + blockSize.z - 1) / blockSize.z);

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla3d_fd_forward", ([&]{
    cuda_nabla3d_fd_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      NZ,
      NY,
      NX,
      hZ,
      Db.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}


torch::Tensor cuda_divergence3d_fd_backward(
  const torch::Tensor &p)
{
  TORCH_CHECK(p.dim() == 4, "Expected 4d tensor");

  const int NZ = p.size(0);
  const int NY = p.size(1);
  const int NX = p.size(2);
  const float hZ = NZ / std::sqrt(0.5*NX*NX+0.5*NY*NY);

  auto divp = torch::zeros({NZ,NY,NX}, p.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y,
                       (NZ + blockSize.z - 1) / blockSize.z  );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence3d_fd_backward", ([&]{
    cuda_divergence3d_fd_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      NZ,
      NY,
      NX,
      hZ,
      divp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}





torch::Tensor cuda_nabla1d_cd_forward(
  const torch::Tensor &b)
{
  TORCH_CHECK(b.dim() == 1, "Expected 1d tensor");

  const int NX = b.size(0);

  auto Db = torch::zeros({NX, 1}, b.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla1d_cd_forward", ([&]{
    cuda_nabla1d_cd_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      NX,
      Db.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}

torch::Tensor cuda_divergence1d_cd_backward(
  const torch::Tensor &p)
{
  TORCH_CHECK(p.dim() == 2, "Expected 2d tensor");

  const int NX = p.size(0);

  auto divp = torch::zeros({NX}, p.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence1d_cd_backward", ([&]{
    cuda_divergence1d_cd_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      p.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      NX,
      divp.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}


torch::Tensor cuda_nabla2d_cd_forward(
  const torch::Tensor &b)
{
  TORCH_CHECK(b.dim() == 2, "Expected 2d tensor");

  const int NY = b.size(0);
  const int NX = b.size(1);

  auto Db = torch::zeros({NY, NX, 2}, b.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                      (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla2d_cd_forward", ([&]{
    cuda_nabla2d_cd_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      NY,
      NX,
      Db.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}


torch::Tensor cuda_divergence2d_cd_backward(
  const torch::Tensor &p)
{
  TORCH_CHECK(p.dim() == 3, "Expected 3d tensor");

  const int NY = p.size(0);
  const int NX = p.size(1);

  auto divp = torch::zeros({NY,NX}, p.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence2d_cd_backward", ([&]{
    cuda_divergence2d_cd_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      p.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      NY,
      NX,
      divp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}

torch::Tensor cuda_nabla3d_cd_forward(
  const torch::Tensor &b)
{
  TORCH_CHECK(b.dim() == 3, "Expected 3d tensor");

  const int NZ = b.size(0);
  const int NY = b.size(1);
  const int NX = b.size(2);
  const float hZ = NZ / std::sqrt(0.5*NX*NX+0.5*NY*NY);

  auto Db = torch::zeros({NZ,NY,NX, 3}, b.options());

  const dim3 blockSize(16,16,3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                      (NY + blockSize.y - 1) / blockSize.y,
                      (NZ + blockSize.z - 1) / blockSize.z);

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla3d_cd_forward", ([&]{
    cuda_nabla3d_cd_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      NZ,
      NY,
      NX,
      hZ,
      Db.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}

torch::Tensor cuda_divergence3d_cd_backward(
  const torch::Tensor &p)
{
  TORCH_CHECK(p.dim() == 4, "Expected 4d tensor");

  const int NZ = p.size(0);
  const int NY = p.size(1);
  const int NX = p.size(2);
  const float hZ = NZ / std::sqrt(0.5*NX*NX+0.5*NY*NY);

  auto divp = torch::zeros({NZ,NY,NX}, p.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y,
                       (NZ + blockSize.z - 1) / blockSize.z  );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence3d_cd_backward", ([&]{
    cuda_divergence3d_cd_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      NZ,
      NY,
      NX,
      hZ,
      divp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}


// torch::Tensor alternative_cuda_nabla3d_cd_forward(
//   const torch::Tensor &b)
// {
//   TORCH_CHECK(b.dim() == 3, "Expected 3d tensor");

//   const int NX = b.size(0);
//   const int NY = b.size(1);
//   const int NZ = b.size(2);
//   const float hZ = NZ / std::sqrt(0.5*NX*NX+0.5*NY*NY);
//   // std::cout << "in cuda_nabla3d_cd_forward:" << std::endl;
//   // std::cout << "(Nx,Ny,Nz) = (" << NX << ", " << NY << "," << NZ << ")" << std::endl;
//   // std::cout << "hZ = " << hZ << std::endl;

//   auto Db = torch::zeros({NX,NY,NZ, 3}, b.options());

//   const dim3 blockSize(16,16,3); 
//   const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
//                       (NY + blockSize.y - 1) / blockSize.y,
//                       (NZ + blockSize.z - 1) / blockSize.z);

// #ifdef CUDA_TIMING
//   CudaTimer cut;
//   cut.start();
// #endif

//   AT_DISPATCH_FLOATING_TYPES(b.type(), "alternative_nabla3d_cd_forward", ([&]{
//     cuda_central_differences_kernel_3D<scalar_t><<<numBlocks, blockSize>>>(
//       b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//       NX,
//       NY,
//       NZ,
//       hZ,
//       Db.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
//   }));
//   cudaSafeCall(cudaGetLastError());

// #ifdef CUDA_TIMING
//   cudaDeviceSynchronize();
//   std::cout << "forward time " << cut.elapsed() << std::endl;
// #endif

//   return Db;
// }