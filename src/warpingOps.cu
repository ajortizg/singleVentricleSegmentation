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
// __forceinline__ __device__ float myabs(const float x)
// {
//   return fabsf(x);
// }

// __forceinline__ __device__ double myabs(const double x)
// {
//   return fabs(x);
// }




// cubic b-splines
// inline __device__ float spline(float x)
// {
//     x = fabsf(x);

//     if (x <= 1.f) return 2.f/3 - (0.5f * x*x) * (2 - x);
//     else if (x <= 2.f) return 1.f/6 * (2-x) * (2-x) * (2-x);
//     else return 0.f;
// }
// inline __device__ double spline(double x)
// {
//     x = fabs(x);

//     if (x <= 1.) return 2./3 - (0.5*x*x) * (2-x);
//     else if (x <= 2.) return 1./6 * (2-x) * (2-x) * (2-x);
//     else return 0.;
// }

// inline __device__ float spline_prime(float x)
// {
//   const float xa = fabsf(x);

//   if (-2.f <= x && x < -1.f) return 0.5f*(2-xa)*(2-xa);
//   else if (-1.f <= x && x < 1.f) return -2*x + 1.5f*xa*x;
//   else if (1.f <= x && x < 2.f) return -0.5f*(2-xa)*(2-xa);
//   else return 0.f;
// }
// inline __device__ double spline_prime(double x)
// {
//   const double xa = fabs(x);
  
//   if (-2. <= x && x < -1.) return 0.5*(2-xa)*(2-xa);
//   else if (-1. <= x && x < 1.) return -2*x + 1.5*xa*x;
//   else if (1. <= x && x < 2.) return -0.5*(2-xa)*(2-xa);
//   else return 0.;
// }

// inline __device__ float spline_prime_prime(float x)
// {
//     x = fabsf(x);

//     if (x <= 1.f) return -2+3*x;
//     else if (x <= 2.f) return 2-x;
//     else return 0.f;
// }
// inline __device__ double spline_prime_prime(double x)
// {
//     x = fabs(x);

//     if (x <= 1.) return -2+3*x;
//     else if (x <= 2.) return 2-x;
//     else return 0.;
// }





/**
 * multilinear interpolation
 */
template <typename T>
__device__ T cuda_interpolate1d_linear(const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, const int NX, const T ix) {
  const int ix_f = floorf(ix);
  const int ix_c = ix_f + 1;
  const T wx = ix - ix_f;

  T u_f = 0;
  if (ix_f >= 0 && ix_f < NX) {
      u_f = u[ix_f];
  }

  T u_c = 0;
  if (ix_c >= 0 && ix_c < NX) {
      u_c = u[ix_c];
  }

  T out = (1 - wx) * u_f;
  out += wx * u_c;

  return out;
}


template <typename T>
__device__ T cuda_interpolate2d_bilinear(const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const T iy, const T ix) {
  const int ix_f = floorf(ix);
  const int ix_c = ix_f + 1;
  const T wx = ix - ix_f;

  const int iy_f = floorf(iy);
  const int iy_c = iy_f + 1;
  const T wy = iy - iy_f;

  T u_ff = 0, u_fc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_ff = u[iy_f][ix_f];

    if (iy_c >= 0 && iy_c < NY)
      u_fc = u[iy_c][ix_f];
  }

  T u_cf = 0, u_cc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_cf = u[iy_f][ix_c];

    if (iy_c >= 0 && iy_c < NY)
      u_cc = u[iy_c][ix_c];
  }

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}


template <typename T>
__device__ T cuda_interpolate3d_trilinear(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const T iz, const T iy, const T ix) {
  const int ix_f = floorf(ix);
  const int ix_c = ix_f + 1;
  const T wx = ix - ix_f;

  const int iy_f = floorf(iy);
  const int iy_c = iy_f + 1;
  const T wy = iy - iy_f;

  const int iz_f = floorf(iz);
  const int iz_c = iz_f + 1;
  const T wz = iz - iz_f;

  T u_fff = 0, u_ffc = 0, u_fcf = 0, u_fcc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fff = u[iz_f][iy_f][ix_f];

        if (iz_c >= 0 && iz_c < NZ)
          u_ffc = u[iz_c][iy_f][ix_f];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fcf = u[iz_f][iy_c][ix_f];

        if (iz_c >= 0 && iz_c < NZ)
          u_fcc = u[iz_c][iy_c][ix_f];
    }
  }

  T u_cff = 0, u_cfc = 0, u_ccf = 0, u_ccc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_cff = u[iz_f][iy_f][ix_c];

        if (iz_c >= 0 && iz_c < NZ)
          u_cfc = u[iz_c][iy_f][ix_c];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_ccf = u[iz_f][iy_c][ix_c];

        if (iz_c >= 0 && iz_c < NZ)
          u_ccc = u[iz_c][iy_c][ix_c];
    }
  }

  T out = (1 - wz) * (1 - wy) * (1 - wx) * u_fff;
  out += wz * (1 - wy) * (1 - wx) * u_ffc;
  out += (1 - wz) * (1 - wy) * wx * u_cff;
  out += wz * (1 - wy) * wx * u_cfc;
  out += (1 - wz) * wy * (1 - wx) * u_fcf;
  out += wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wy * wx * u_ccc;

  return out;
}




/**
 * cubic spline interpolation
 */

template <typename T>
__device__ T cuda_interpolate1d_cubicSpline_local(volatile T* localBuffer, const T local_ix) {

  const int kernel_size=4;

  const int ix_f = floorf(local_ix);
  const int ix_c = ix_f + 1;
  const int ix_f_1 = ix_f - 1;
  const int ix_c_1 = ix_c + 1;

  // get the input values
  T i_f = 0;
  if (ix_f >= 0 && ix_f < kernel_size) i_f = localBuffer[ix_f];
  T i_f_1 = 0;
  if (ix_f_1 >= 0 && ix_f_1 < kernel_size) i_f_1 = localBuffer[ix_f_1];
  T i_c = 0;
  if (ix_c >= 0 && ix_c < kernel_size) i_c = localBuffer[ix_c];
  T i_c_1 = 0;
  if (ix_c_1 >= 0 && ix_c_1 < kernel_size) i_c_1 = localBuffer[ix_c_1];

  // determine the coefficients
  const T p_f = i_f;
  const T p_prime_f = (i_c - i_f_1) / 2;
  const T p_c = i_c;
  const T p_prime_c = (i_c_1 - i_f) / 2;

  const T a = 2 * p_f - 2 * p_c + p_prime_f + p_prime_c;
  const T b = -3 * p_f + 3 * p_c - 2 * p_prime_f - p_prime_c;
  const T c = p_prime_f;
  const T d = p_f;

  const T wx = local_ix - ix_f;

  T out = wx * (wx * (wx * a + b) + wx) + d;

  return out;
}


template <typename T>
__device__ T cuda_interpolate1d_cubic(const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, const int NX, const T ix) {

  const int ix_f = floorf(ix);
  const T wx = ix - ix_f;
  T buff_x[4];

  for (int dx = -1; dx < 3; ++dx)
  {
        const int c_ix_x = ix_f + dx;
        if (c_ix_x >= 0 && c_ix_x < NX)
          buff_x[dx + 1] = u[c_ix_x];
        else
          buff_x[dx + 1] = 0;
  }

  T out = cuda_interpolate1d_cubicSpline_local<T>(buff_x, wx + 1);

  return out;
}


template <typename T>
__device__ T cuda_interpolate2d_bicubic(const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u, 
                                      const int NY, const int NX, const T iy, const T ix) {

  const int ix_f = floorf(ix);
  const T wx = ix - ix_f;

  const int iy_f = floorf(iy);
  const T wy = iy - iy_f;

  T buff_y[4];
  T buff_x[4];

  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_ix_y = iy_f + dy;

    if (c_ix_y >= 0 && c_ix_y < NY)
    {
      // get the input values
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_ix_x = ix_f + dx;
        if (c_ix_x >= 0 && c_ix_x < NX)
          buff_x[dx + 1] = u[c_ix_y][c_ix_x];
        else
          buff_x[dx + 1] = 0;
      }
      buff_y[dy + 1] = cuda_interpolate1d_cubicSpline_local<T>(buff_x, wx + 1);
    }
    else
      buff_y[dy + 1] = 0;
  }

  T out = cuda_interpolate1d_cubicSpline_local<T>(buff_y, wy + 1);

  return out;
}










// CUDA kernels

template <typename T>
__global__ void cuda_warp1d_cubicSpline_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> phi,
  const int NX,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
    const T dx = phi[ix][0];
    u_warped[ix] = cuda_interpolate1d_linear(u, NX, ix + dx);
    //u_warped[ix] = cuda_interpolate1d_bicubic(u, idx + dx);
  }
  
}

template <typename T>
__global__ void cuda_warp2d_cubicSpline_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> phi,
  const int NY,
  const int NX,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u_warped)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    const T dx = phi[iy][ix][0];
    const T dy = phi[iy][ix][1];
    //u_warped[iy][ix] = cuda_interpolate2d_bilinear(u, NY, NX, iy + dy, ix + dx);
    u_warped[iy][ix] = cuda_interpolate2d_bicubic(u, NY, NX, iy + dy, ix + dx);
  }

}



template <typename T>
__global__ void cuda_warp3d_cubicSpline_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> phi,
  const int NZ,
  const int NY,
  const int NX,
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
    u_warped[iz][iy][ix] = cuda_interpolate3d_trilinear(u, NZ, NY, NX, iz + dz, iy + dy, ix + dx);
    //u_warped[ix] = cuda_interpolate1d_bicubic(u, ix + dx);
  }

}

// ======================================================
// C++ kernel calls
// ======================================================
torch::Tensor cuda_warp1d_cubicSpline(
  const torch::Tensor &u,
  const torch::Tensor &phi )
{
  TORCH_CHECK(u.dim() == 1, "Expected 1d tensor");
  TORCH_CHECK(phi.dim() == 2, "Expected 2d tensor")

  const int NX = u.size(0);

  auto u_warped = torch::zeros({NX}, u.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(u.type(), "warp1d_cubicSpline", ([&]{
    cuda_warp1d_cubicSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
      u.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      phi.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      NX,
      u_warped.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}


torch::Tensor cuda_warp2d_cubicSpline(
  const torch::Tensor &u,
  const torch::Tensor &phi )
{
  TORCH_CHECK(u.dim() == 2, "Expected 2d tensor");
  TORCH_CHECK(phi.dim() == 3, "Expected 3d tensor")

  const int NY = u.size(0);
  const int NX = u.size(1);

  auto u_warped = torch::zeros({NY,NX}, u.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(u.type(), "warp2d_cubicSpline", ([&]{
    cuda_warp2d_cubicSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
      u.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      phi.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      NY, NX,
      u_warped.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return u_warped;
}





torch::Tensor cuda_warp3d_cubicSpline(
  const torch::Tensor &u,
  const torch::Tensor &phi )
{
  TORCH_CHECK(u.dim() == 3, "Expected 3d tensor");
  TORCH_CHECK(phi.dim() == 4, "Expected 4d tensor")

  const int NZ = u.size(0);
  const int NY = u.size(1);
  const int NX = u.size(2);

  auto u_warped = torch::zeros({NZ,NY,NX}, u.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y, (NZ + blockSize.z - 1) / blockSize.z );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  AT_DISPATCH_FLOATING_TYPES(u.type(), "warp3d_cubicSpline", ([&]{
    cuda_warp3d_cubicSpline_kernel<scalar_t><<<numBlocks, blockSize>>>(
      u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      phi.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      NZ, NY, NX,
      u_warped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));
  cudaSafeCall(cudaGetLastError());

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