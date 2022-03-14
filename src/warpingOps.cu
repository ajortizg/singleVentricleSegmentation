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
__global__ void cuda_warp1d_linear_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> phi,
  const int NX, const float LX, const float hX,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
    const T dx = phi[ix][0];
    const T coord_x_warped = ix * hX + dx;
    u_warped[ix] = cuda_interpolate1d_linear(u, NX, LX, hX, coord_x_warped );
  }
  
}

template <typename T>
__global__ void cuda_warp2d_bilinear_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
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
    u_warped[iy][ix] = cuda_interpolate2d_bilinear(u, NY, NX, LY, LX, hY, hX, coord_y_warped, coord_x_warped);
  }

}

template <typename T>
__global__ void cuda_warp3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
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
    u_warped[iz][iy][ix] = cuda_interpolate3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, coord_z_warped, coord_y_warped, coord_x_warped);
  }
}

template <typename T>
__global__ void cuda_warpVectorField3d_trilinear_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
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
    for(int comp=0; comp<3; ++comp)
    {
      u_warped[iz][iy][ix][comp] = cuda_interpolateVectorField3d_trilinear(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, coord_z_warped, coord_y_warped, coord_x_warped, comp);
    }
  }
}


template <typename T>
__global__ void cuda_warp1d_cubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> phi,
  const int NX, const float LX, const float hX,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
    const T dx = phi[ix][0];
    const T coord_x_warped = ix * hX + dx;
    u_warped[ix] = cuda_interpolate1d_cubicHermiteSpline(u, NX, LX, hX, coord_x_warped);
  }
  
}

template <typename T>
__global__ void cuda_warp2d_bicubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const int NY, const int NX,
  const float LY, const float LX,
  const float hY, const float hX,
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
    u_warped[iy][ix] = cuda_interpolate2d_bicubicHermiteSpline(u, NY, NX, LY, LX, hY, hX, coord_y_warped, coord_x_warped);
  }

}

template <typename T>
__global__ void cuda_warp3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
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
    u_warped[iz][iy][ix] = cuda_interpolate3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, coord_z_warped, coord_y_warped, coord_x_warped);
  }
}

template <typename T>
__global__ void cuda_warpVectorField3d_tricubicHermiteSpline_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ, const int NY, const int NX,
  const float LZ, const float LY, const float LX,
  const float hZ, const float hY, const float hX,
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
    for(int comp=0; comp<3; ++comp )
    {
      u_warped[iz][iy][ix][comp] = cuda_interpolateVectorField3d_tricubicHermiteSpline(u, NZ, NY, NX, LZ, LY, LX, hZ, hY, hX, coord_z_warped, coord_y_warped, coord_x_warped, comp);
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
  const InterpolationType interpolation = INTERPOLATE_LINEAR )
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
  case INTERPOLATE_LINEAR: // fallthrough intended
    AT_DISPATCH_FLOATING_TYPES(u.type(), "warp1d_linear", ([&]{
      cuda_warp1d_linear_kernel<scalar_t><<<numBlocks, blockSize>>>(
        u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        phi.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        NX, LX, hX,
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
        u_warped.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;

  }


#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}


torch::Tensor cuda_warp2d(
  const torch::Tensor &u,
  const torch::Tensor &phi,
  const MeshInfo2D& meshInfo,
  const InterpolationType interpolation = INTERPOLATE_LINEAR )
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
  case INTERPOLATE_LINEAR: // fallthrough intended
    AT_DISPATCH_FLOATING_TYPES(u.type(), "warp2d_bilinear", ([&]{
      cuda_warp2d_bilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
        u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        NY, NX,
        LY, LX,
        hY, hX,
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
        u_warped.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;

  }

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
  const InterpolationType interpolation = INTERPOLATE_LINEAR )
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
  case INTERPOLATE_LINEAR: // fallthrough intended
    AT_DISPATCH_FLOATING_TYPES(u.type(), "warp3d_trilinear", ([&]{
      cuda_warp3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
        u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        NZ, NY, NX,
        LZ, LY, LX,
        hZ, hY, hX,
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
        u_warped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}



torch::Tensor cuda_warpVectorField3d(
  const torch::Tensor &u,
  const torch::Tensor &phi,
  const MeshInfo3D& meshInfo,
  const InterpolationType interpolation = INTERPOLATE_LINEAR )
{
  TORCH_CHECK(u.dim() == 4, "Expected 4d tensor");
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

  auto u_warped = torch::zeros({NZ,NY,NX,3}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(interpolation)
  {
  case INTERPOLATE_LINEAR: // fallthrough intended
    AT_DISPATCH_FLOATING_TYPES(u.type(), "warpVectorField3d_trilinear", ([&]{
      cuda_warpVectorField3d_trilinear_kernel<scalar_t><<<numBlocks, blockSize>>>(
        u.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        NZ, NY, NX,
        LZ, LY, LX,
        hZ, hY, hX,
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
        u_warped.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}







//============================================================================
// gradient operator
//============================================================================
/**
  * perform bilinear interpolation adjoint
  */
// template <typename T>
// __device__ T backpolate_bilinear(typename Tensor5<T>::Tensor& grad_x,
//                                  T& grad_idx, T& grad_idy,
//                                  const typename Tensor5<T>::ConstTensor& x,
//                                  T val, int ids, int idc, T idy, T idx, int idalpha) {
//   const int idx_f = floorf(idx);
//   const int idy_f = floorf(idy);

//   const int idx_c = idx_f + 1;
//   const int idy_c = idy_f + 1;

//   const T w = idx - idx_f;
//   const T h = idy - idy_f;

//   T i_ff = 0, i_fc = 0;
//   if (idx_f >= 0 && idx_f < grad_x.dimensions()[4]) {
//     if (idy_f >= 0 && idy_f < grad_x.dimensions()[3]) {
//       tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, idy_f, idx_f),
//                                 (1 - h) * (1 - w) * val);
//       i_ff = x(ids, idc, idalpha, idy_f, idx_f);
//     }

//     if (idy_c >= 0 && idy_c < grad_x.dimensions()[3]) {
//       tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, idy_c, idx_f),
//                                 h * (1 - w) * val);
//       i_fc = x(ids, idc, idalpha, idy_c, idx_f);
//     }
//   }

//   T i_cf = 0, i_cc = 0;
//   if (idx_c >= 0 && idx_c < grad_x.dimensions()[4]) {
//     if (idy_f >= 0 && idy_f < grad_x.dimensions()[3]) {
//       tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, idy_f, idx_c),
//                                 (1 - h) * w * val);
//       i_cf = x(ids, idc, idalpha, idy_f, idx_c);
//     }

//     if (idy_c >= 0 && idy_c < grad_x.dimensions()[3]) {
//       tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, idy_c, idx_c),
//                                 h * w * val);
//       i_cc = x(ids, idc, idalpha, idy_c, idx_c);
//     }
//   }

//   grad_idx += ((1 - h) * (i_cf - i_ff) + h * (i_cc - i_fc)) * val;
//   grad_idy += ((1 - w) * (i_fc - i_ff) + w * (i_cc - i_cf)) * val;

//   T out = (1 - h) * (1 - w) * i_ff;
//   out += (1 - h) * w * i_cf;
//   out += h * (1 - w) * i_fc;
//   out += h * w * i_cc;

//   return out;
// }

/**
 * perform cubic interpolation on the input image i_in given the
 * index (idx,idy)
 */
// template<typename T>
// __device__ void backpolate_cubic(volatile T *grad_x, T& grad_idx,
//    T* in, T error, T idx, int kernel_size)
// {
//   const int idx_f = floorf(idx);
//   const int idx_f_1 = idx_f - 1;
//   const int idx_c = idx_f+1;
//   const int idx_c_1 = idx_c+1;

//   const T u = idx - idx_f;
//   const T uu = u*u;
//   const T uuu = uu*u;

//   // determine the coefficients
//   T d_out_d_p_f_1 = -uuu/2 + uu - u/2;
//   T d_out_d_p_f = (3*uuu)/2 - (5*uu)/2 + 1;
//   T d_out_d_p_c = -(3*uuu)/2 + 2*uu + u/2;
//   T d_out_d_p_c_1 = uuu/2 - uu/2;

//   T i_f = 0;
//   if (idx_f >= 0 && idx_f < kernel_size)
//   {
//     i_f = in[idx_f];
//     grad_x[idx_f] = d_out_d_p_f   * error;
//   }
//   else
//     grad_x[idx_f] = 0;

//   T i_f_1 = 0;
//   if (idx_f_1 >= 0 && idx_f_1 < kernel_size)
//   {
//     i_f_1 = in[idx_f_1];
//     grad_x[idx_f_1] = d_out_d_p_f_1 * error;
//   }
//   else
//     grad_x[idx_f_1] = 0;

//   T i_c = 0;
//   if (idx_c >= 0 && idx_c < kernel_size)
//   {
//     i_c = in[idx_c];
//     grad_x[idx_c] = d_out_d_p_c   * error;
//   }
//   else
//     grad_x[idx_c] = 0;

//   T i_c_1 = 0;
//   if (idx_c_1 >= 0 && idx_c_1 < kernel_size)
//   {
//     i_c_1 = in[idx_c_1];
//     grad_x[idx_c_1] = d_out_d_p_c_1 * error;
//   }
//   else
//     grad_x[idx_c_1] = 0;

//   // determine the coefficients
//   const T p_f = i_f;
//   const T p_prime_f = (i_c - i_f_1) / 2;
//   const T p_c = i_c;
//   const T p_prime_c = (i_c_1 - i_f) / 2;

//   const T a = 2 * p_f - 2 * p_c + p_prime_f + p_prime_c;
//   const T b = -3 * p_f + 3 * p_c - 2 * p_prime_f - p_prime_c;
//   const T c = p_prime_f;

//   grad_idx += (3*uu*a + 2*b*u + c) * error;
// }

// template<typename T>
// __device__ T backpolate_bicubic(typename Tensor5<T>::Tensor& grad_x,
//   T& grad_idx, T& grad_idy,
//   const typename Tensor5<T>::ConstTensor& x,
//   T val, int ids, int idc, T idy, T idx, int idalpha)
// {
//   const int idy_f = floorf(idy);
//   const int idx_f = floorf(idx);

//   T buff_y[4];
//   T buff_x[4];

//   // first perform interpolation
//   for (int dy = -1; dy < 3; ++dy)
//   {
//     const int c_idx_y = idy_f + dy;

//     if (c_idx_y >= 0 && c_idx_y < x.dimensions()[3])
//     {
//       // get the input values
//       for (int dx = -1; dx < 3; ++dx)
//       {
//         const int c_idx_x = idx_f + dx;
//         if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
//           buff_x[dx + 1] = x(ids, idc, idalpha, c_idx_y, c_idx_x);
//         else
//           buff_x[dx + 1] = 0;
//       }
//       buff_y[dy + 1] = interpolate_cubic<T>(buff_x, idx - idx_f + 1, 4);
//     }
//     else
//       buff_y[dy + 1] = 0;
//   }

//   T out = interpolate_cubic<T>(buff_y, idy - idy_f + 1, 4);

//   // backpolate the error
//   T buff_grad_y[4];
//   backpolate_cubic<T>(buff_grad_y, grad_idy, buff_y, val, idy - idy_f + 1, 4);

//   T buff_grad_x[4];
//   for (int dy = -1; dy < 3; ++dy)
//   {
//     const int c_idx_y = idy_f + dy;

//     if (c_idx_y >= 0 && c_idx_y < x.dimensions()[3])
//     {
//       // get the input values
//       for (int dx = -1; dx < 3; ++dx)
//       {
//         const int c_idx_x = idx_f + dx;
//         if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
//           buff_x[dx + 1] = x(ids, idc, idalpha, c_idx_y, c_idx_x);
//         else
//           buff_x[dx + 1] = 0;
//       }
//       backpolate_cubic<T>(buff_grad_x, grad_idx, buff_x, buff_grad_y[dy+1], idx - idx_f + 1, 4);
//       for (int dx = -1; dx < 3; ++dx)
//       {
//         const int c_idx_x = idx_f + dx;
//         if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
//           tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, c_idx_y, c_idx_x),
//                                     buff_grad_x[dx + 1]);
//       }
//     }
//   }
//   return out;
// }



// template <typename T>
// __device__ T backpolate_bilinear(typename Tensor4<T>::Tensor& grad_x,
//                                  T& grad_idx, T& grad_idy,
//                                  const typename Tensor4<T>::ConstTensor& x,
//                                  T val, int ids, int idc, T idy, T idx) {
//   const int idx_f = floorf(idx);
//   const int idy_f = floorf(idy);

//   const int idx_c = idx_f + 1;
//   const int idy_c = idy_f + 1;

//   const T w = idx - idx_f;
//   const T h = idy - idy_f;

//   T i_ff = 0, i_fc = 0;
//   if (idx_f >= 0 && idx_f < grad_x.dimensions()[3]) {
//     if (idy_f >= 0 && idy_f < grad_x.dimensions()[2]) {
//       tficg::CudaAtomicAdd(&grad_x(ids, idc, idy_f, idx_f),
//                                 (1 - h) * (1 - w) * val);
//       i_ff = x(ids, idc, idy_f, idx_f);
//     }

//     if (idy_c >= 0 && idy_c < grad_x.dimensions()[2]) {
//       tficg::CudaAtomicAdd(&grad_x(ids, idc, idy_c, idx_f),
//                                 h * (1 - w) * val);
//       i_fc = x(ids, idc, idy_c, idx_f);
//     }
//   }

//   T i_cf = 0, i_cc = 0;
//   if (idx_c >= 0 && idx_c < grad_x.dimensions()[3]) {
//     if (idy_f >= 0 && idy_f < grad_x.dimensions()[2]) {
//       tficg::CudaAtomicAdd(&grad_x(ids, idc, idy_f, idx_c),
//                                 (1 - h) * w * val);
//       i_cf = x(ids, idc, idy_f, idx_c);
//     }

//     if (idy_c >= 0 && idy_c < grad_x.dimensions()[2]) {
//       tficg::CudaAtomicAdd(&grad_x(ids, idc, idy_c, idx_c),
//                                 h * w * val);
//       i_cc = x(ids, idc, idy_c, idx_c);
//     }
//   }

//   grad_idx += ((1 - h) * (i_cf - i_ff) + h * (i_cc - i_fc)) * val;
//   grad_idy += ((1 - w) * (i_fc - i_ff) + w * (i_cc - i_cf)) * val;

//   T out = (1 - h) * (1 - w) * i_ff;
//   out += (1 - h) * w * i_cf;
//   out += h * (1 - w) * i_fc;
//   out += h * w * i_cc;

//   return out;
// }


// template<typename T>
// __device__ T backpolate_bicubic(typename Tensor4<T>::Tensor& grad_x,
//   T& grad_idx, T& grad_idy,
//   const typename Tensor4<T>::ConstTensor& x,
//   T val, int ids, int idc, T idy, T idx)
// {
//   const int idy_f = floorf(idy);
//   const int idx_f = floorf(idx);

//   T buff_y[4];
//   T buff_x[4];

//   // first perform interpolation
//   for (int dy = -1; dy < 3; ++dy)
//   {
//     const int c_idx_y = idy_f + dy;

//     if (c_idx_y >= 0 && c_idx_y < x.dimensions()[2])
//     {
//       // get the input values
//       for (int dx = -1; dx < 3; ++dx)
//       {
//         const int c_idx_x = idx_f + dx;
//         if (c_idx_x >= 0 && c_idx_x < x.dimensions()[3])
//           buff_x[dx + 1] = x(ids, idc, c_idx_y, c_idx_x);
//         else
//           buff_x[dx + 1] = 0;
//       }
//       buff_y[dy + 1] = interpolate_cubic<T>(buff_x, idx - idx_f + 1, 4);
//     }
//     else
//       buff_y[dy + 1] = 0;
//   }

//   T out = interpolate_cubic<T>(buff_y, idy - idy_f + 1, 4);

//   // backpolate the error
//   T buff_grad_y[4];
//   backpolate_cubic<T>(buff_grad_y, grad_idy, buff_y, val, idy - idy_f + 1, 4);

//   T buff_grad_x[4];
//   for (int dy = -1; dy < 3; ++dy)
//   {
//     const int c_idx_y = idy_f + dy;

//     if (c_idx_y >= 0 && c_idx_y < x.dimensions()[2])
//     {
//       // get the input values
//       for (int dx = -1; dx < 3; ++dx)
//       {
//         const int c_idx_x = idx_f + dx;
//         if (c_idx_x >= 0 && c_idx_x < x.dimensions()[3])
//           buff_x[dx + 1] = x(ids, idc, c_idx_y, c_idx_x);
//         else
//           buff_x[dx + 1] = 0;
//       }
//       backpolate_cubic<T>(buff_grad_x, grad_idx, buff_x, buff_grad_y[dy+1], idx - idx_f + 1, 4);
//       for (int dx = -1; dx < 3; ++dx)
//       {
//         const int c_idx_x = idx_f + dx;
//         if (c_idx_x >= 0 && c_idx_x < x.dimensions()[3])
//           tficg::CudaAtomicAdd(&grad_x(ids, idc, c_idx_y, c_idx_x),
//                                     buff_grad_x[dx + 1]);
//       }
//     }
//   }
//   return out;
// }