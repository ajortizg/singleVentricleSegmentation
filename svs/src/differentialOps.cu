#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 
#include <iostream>
#include <stdio.h>

#include "coreDefines.h"
#include "boundary.cuh"
#include "cudaDebug.cuh"


//=========================================================
// CUDA kernels
//=========================================================

template <typename T>
__global__ void cuda_nabla1d_cd_forward_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> b,
  const int NX, const float hX,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
      Db[ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[ix+1] - b[ix-1])/hX
                              : 
                              0.5*(b[ix] - b[ix-1])/hX
                            : 
                            0.5*(b[ix+1] - b[ix])/hX;
  }
}


template <typename T>
__global__ void cuda_nabla1d_cd_forward_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> b,
  const int NX, const float hX,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
      Db[ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[ix+1] - b[ix-1])/hX
                              : 
                              0.
                            : 
                            0.;
  }
}


template <typename T>
__global__ void cuda_nabla1d_cd_forward_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> b,
  const int NX, const float hX,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
      Db[ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[ix+1] - b[ix-1])/hX
                              : 
                              0.5*(b[ix] - b[ix-1])/hX
                            : 
                            0.5*(b[ix+1] - b[ix])/hX;
  }
}


template <typename T>
__global__ void cuda_divergence1d_cd_backward_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> p,
  const int NX, const float hX,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               0.5*(p[ix-1][0] - p[ix+1][0]) 
                               : 
                               0.5*(p[ix-1][0] + p[ix][0]) 
                              :
                              0.5*(-p[ix][0] - p[ix+1][0]);
      divp[ix] = divp_x/hX;
  }
}

template <typename T>
__global__ void cuda_divergence1d_cd_backward_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> p,
  const int NX, const float hX,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {

      T divp_x = (ix > 1) ? 
                            (ix < NX - 2 ) ? 
                               0.5*(p[ix-1][0] - p[ix+1][0]) 
                               : 
                               0.5*p[ix-1][0] 
                              :
                              -0.5*p[ix+1][0];

      divp[ix] = divp_x/hX;
  }
}

template <typename T>
__global__ void cuda_divergence1d_cd_backward_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> p,
  const int NX, const float hX,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix < NX )
  {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               0.5*(p[ix-1][0] - p[ix+1][0]) 
                               : 
                               0.5*(p[ix-1][0] + p[ix][0]) 
                              :
                              0.5*(-p[ix][0] - p[ix+1][0]);
      divp[ix] = divp_x/hX;
  }
}


template <typename T>
__global__ void cuda_nabla2d_cd_forward_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> b,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
      Db[iy][ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iy][ix+1] - b[iy][ix-1])/hX
                              : 
                              0.5*(b[iy][ix] - b[iy][ix-1])/hX
                            : 
                            0.5*(b[iy][ix+1] - b[iy][ix])/hX;

      Db[iy][ix][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iy+1][ix] - b[iy-1][ix])/hY
                              : 
                              0.5*(b[iy][ix] - b[iy-1][ix])/hY
                            : 
                            0.5*(b[iy+1][ix] - b[iy][ix])/hY;
  }
}

template <typename T>
__global__ void cuda_nabla2d_cd_forward_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> b,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
      Db[iy][ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iy][ix+1] - b[iy][ix-1])/hX
                              : 
                              0.
                            : 
                            0.;

      Db[iy][ix][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iy+1][ix] - b[iy-1][ix])/hY
                              : 
                              0.
                            : 
                            0.;
  }
}

template <typename T>
__global__ void cuda_nabla2d_cd_forward_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> b,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
      Db[iy][ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iy][ix+1] - b[iy][ix-1])/hX
                              : 
                              0.5*(b[iy][ix] - b[iy][ix-1])/hX
                            : 
                            0.5*(b[iy][ix+1] - b[iy][ix])/hX;

      Db[iy][ix][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iy+1][ix] - b[iy-1][ix])/hY
                              : 
                              0.5*(b[iy][ix] - b[iy-1][ix])/hY
                            : 
                            0.5*(b[iy+1][ix] - b[iy][ix])/hY;
  }
}


template <typename T>
__global__ void cuda_divergence2d_cd_backward_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> p,
  const int NY, const int NX,
  const float hY, const float hX,
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

      divp[iy][ix] = divp_x/hX + divp_y/hY;
  }
}

template <typename T>
__global__ void cuda_divergence2d_cd_backward_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> p,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
      T divp_x = (ix > 1) ? 
                            (ix < NX - 2 ) ? 
                               0.5*(p[iy][ix-1][0] - p[iy][ix+1][0]) 
                               : 
                               0.5*p[iy][ix-1][0]
                              :
                              -0.5*p[iy][ix+1][0];

      T divp_y = (iy > 1) ? 
                            (iy < NY - 2 ) ? 
                               0.5*(p[iy-1][ix][1] - p[iy+1][ix][1]) 
                               : 
                               0.5*p[iy-1][ix][1]
                              :
                              -0.5*p[iy+1][ix][1];

      divp[iy][ix] = divp_x/hX + divp_y/hY;
  }
}

template <typename T>
__global__ void cuda_divergence2d_cd_backward_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> p,
  const int NY, const int NX,
  const float hY, const float hX,
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

      divp[iy][ix] = divp_x/hX + divp_y/hY;
  }
}

template <typename T>
__global__ void cuda_nabla2d_cd_forwardVectorField_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> b,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    for(int comp=0; comp<2; ++comp )
    {
      Db[iy][ix][comp][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iy][ix+1][comp] - b[iy][ix-1][comp])/hX
                              : 
                              0.5*(b[iy][ix][comp] - b[iy][ix-1][comp])/hX
                            : 
                            0.5*(b[iy][ix+1][comp] - b[iy][ix][comp])/hX;

      Db[iy][ix][comp][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iy+1][ix][comp] - b[iy-1][ix][comp])/hY
                              : 
                              0.5*(b[iy][ix][comp] - b[iy-1][ix][comp])/hY
                            : 
                            0.5*(b[iy+1][ix][comp] - b[iy][ix][comp])/hY;
    }
  }
}

template <typename T>
__global__ void cuda_nabla2d_cd_forwardVectorField_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> b,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    for(int comp=0; comp<2; ++comp )
    {
      Db[iy][ix][comp][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iy][ix+1][comp] - b[iy][ix-1][comp])/hX
                              : 
                              0.
                            : 
                            0.;

      Db[iy][ix][comp][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iy+1][ix][comp] - b[iy-1][ix][comp])/hY
                              : 
                              0.
                            : 
                            0.;
    }
  }
}

template <typename T>
__global__ void cuda_nabla2d_cd_forwardVectorField_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> b,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    for(int comp=0; comp<2; ++comp )
    {
      Db[iy][ix][comp][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iy][ix+1][comp] - b[iy][ix-1][comp])/hX
                              : 
                              0.5*(b[iy][ix][comp] - b[iy][ix-1][comp])/hX
                            : 
                            0.5*(b[iy][ix+1][comp] - b[iy][ix][comp])/hX;

      Db[iy][ix][comp][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iy+1][ix][comp] - b[iy-1][ix][comp])/hY
                              : 
                              0.5*(b[iy][ix][comp] - b[iy-1][ix][comp])/hY
                            : 
                            0.5*(b[iy+1][ix][comp] - b[iy][ix][comp])/hY;
    }
  }
}


template <typename T>
__global__ void cuda_divergence2d_cd_backwardVectorField_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    for(int comp=0; comp<2; ++comp )
    {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               0.5*(p[iy][ix-1][comp][0] - p[iy][ix+1][comp][0]) 
                               : 
                               0.5*(p[iy][ix-1][comp][0] + p[iy][ix][comp][0]) 
                              :
                              0.5*(-p[iy][ix][comp][0] - p[iy][ix+1][comp][0]);

      T divp_y = (iy > 0) ? 
                            (iy < NY - 1 ) ? 
                               0.5*(p[iy-1][ix][comp][1] - p[iy+1][ix][comp][1]) 
                               : 
                               0.5*(p[iy-1][ix][comp][1] + p[iy][ix][comp][1]) 
                              :
                              0.5*(-p[iy][ix][comp][1] - p[iy+1][ix][comp][1]);

      divp[iy][ix][comp] = divp_x/hX + divp_y/hY;
    }
  }
}

template <typename T>
__global__ void cuda_divergence2d_cd_backwardVectorField_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    for(int comp=0; comp<2; ++comp )
    {
      T divp_x = (ix > 1) ? 
                            (ix < NX - 2 ) ? 
                               0.5*(p[iy][ix-1][comp][0] - p[iy][ix+1][comp][0]) 
                               : 
                               0.5*p[iy][ix-1][comp][0] 
                              :
                              -0.5*p[iy][ix+1][comp][0];

      T divp_y = (iy > 1) ? 
                            (iy < NY - 2 ) ? 
                               0.5*(p[iy-1][ix][comp][1] - p[iy+1][ix][comp][1]) 
                               : 
                               0.5*p[iy-1][ix][comp][1] 
                              :
                              -0.5*p[iy+1][ix][comp][1];

      divp[iy][ix][comp] = divp_x/hX + divp_y/hY;
    }
  }
}

template <typename T>
__global__ void cuda_divergence2d_cd_backwardVectorField_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
  const int NY, const int NX,
  const float hY, const float hX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < NX && iy < NY )
  {
    for(int comp=0; comp<2; ++comp )
    {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               0.5*(p[iy][ix-1][comp][0] - p[iy][ix+1][comp][0]) 
                               : 
                               0.5*(p[iy][ix-1][comp][0] + p[iy][ix][comp][0]) 
                              :
                              0.5*(-p[iy][ix][comp][0] - p[iy][ix+1][comp][0]);

      T divp_y = (iy > 0) ? 
                            (iy < NY - 1 ) ? 
                               0.5*(p[iy-1][ix][comp][1] - p[iy+1][ix][comp][1]) 
                               : 
                               0.5*(p[iy-1][ix][comp][1] + p[iy][ix][comp][1]) 
                              :
                              0.5*(-p[iy][ix][comp][1] - p[iy+1][ix][comp][1]);

      divp[iy][ix][comp] = divp_x/hX + divp_y/hY;
    }
  }
}

template <typename T>
__global__ void cuda_nabla3d_cd_forward_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> b,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
      Db[iz][iy][ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iz][iy][ix+1] - b[iz][iy][ix-1])/hX
                              : 
                              0.5*(b[iz][iy][ix] - b[iz][iy][ix-1])/hX
                            : 
                            0.5*(b[iz][iy][ix+1] - b[iz][iy][ix])/hX;

      Db[iz][iy][ix][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iz][iy+1][ix] - b[iz][iy-1][ix])/hY
                              : 
                              0.5*(b[iz][iy][ix] - b[iz][iy-1][ix])/hY
                            : 
                            0.5*(b[iz][iy+1][ix] - b[iz][iy][ix])/hY;

      Db[iz][iy][ix][2] = (iz > 0) ? 
                            (iz < NZ-1) ? 
                              0.5*(b[iz+1][iy][ix] - b[iz-1][iy][ix])/hZ
                              : 
                              0.5*(b[iz][iy][ix] - b[iz-1][iy][ix])/hZ
                            : 
                            0.5*(b[iz+1][iy][ix] - b[iz][iy][ix])/hZ;
  }
}

template <typename T>
__global__ void cuda_nabla3d_cd_forward_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> b,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
      Db[iz][iy][ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iz][iy][ix+1] - b[iz][iy][ix-1])/hX
                              : 
                              0.
                            : 
                            0.;

      Db[iz][iy][ix][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iz][iy+1][ix] - b[iz][iy-1][ix])/hY
                              : 
                              0.
                            : 
                            0.;

      Db[iz][iy][ix][2] = (iz > 0) ? 
                            (iz < NZ-1) ? 
                              0.5*(b[iz+1][iy][ix] - b[iz-1][iy][ix])/hZ
                              : 
                              0.
                            : 
                            0.;
  }
}


template <typename T>
__global__ void cuda_nabla3d_cd_forward_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> b,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
      Db[iz][iy][ix][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iz][iy][ix+1] - b[iz][iy][ix-1])/hX
                              : 
                              0.5*(b[iz][iy][ix] - b[iz][iy][ix-1])/hX
                            : 
                            0.5*(b[iz][iy][ix+1] - b[iz][iy][ix])/hX;

      Db[iz][iy][ix][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iz][iy+1][ix] - b[iz][iy-1][ix])/hY
                              : 
                              0.5*(b[iz][iy][ix] - b[iz][iy-1][ix])/hY
                            : 
                            0.5*(b[iz][iy+1][ix] - b[iz][iy][ix])/hY;

      Db[iz][iy][ix][2] = (iz > 0) ? 
                            (iz < NZ-1) ? 
                              0.5*(b[iz+1][iy][ix] - b[iz-1][iy][ix])/hZ
                              : 
                              0.5*(b[iz][iy][ix] - b[iz-1][iy][ix])/hZ
                            : 
                            0.5*(b[iz+1][iy][ix] - b[iz][iy][ix])/hZ;
  }
}

template <typename T>
__global__ void cuda_divergence3d_cd_backward_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
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

      divp[iz][iy][ix] = divp_x/hX + divp_y/hY + divp_z/hZ;
  }
}

template <typename T>
__global__ void cuda_divergence3d_cd_backward_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
      T divp_x = (ix > 1) ? 
                            (ix < NX - 2 ) ? 
                               0.5*(p[iz][iy][ix-1][0] - p[iz][iy][ix+1][0]) 
                               : 
                               0.5*p[iz][iy][ix-1][0]
                              :
                              -0.5*p[iz][iy][ix+1][0];

      T divp_y = (iy > 1) ? 
                            (iy < NY - 2 ) ? 
                               0.5*(p[iz][iy-1][ix][1] - p[iz][iy+1][ix][1]) 
                               : 
                               0.5*p[iz][iy-1][ix][1]
                              :
                              -0.5*p[iz][iy+1][ix][1];

      T divp_z = (iz > 1) ? 
                            (iz < NZ - 2 ) ? 
                               0.5*(p[iz-1][iy][ix][2] - p[iz+1][iy][ix][2]) 
                               : 
                               0.5*p[iz-1][iy][ix][2]
                              :
                              -0.5*p[iz+1][iy][ix][2];

      divp[iz][iy][ix] = divp_x/hX + divp_y/hY + divp_z/hZ;
  }
}


template <typename T>
__global__ void cuda_divergence3d_cd_backward_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> p,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
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

      divp[iz][iy][ix] = divp_x/hX + divp_y/hY + divp_z/hZ;
  }
}

template <typename T>
__global__ void cuda_nabla3d_cd_forwardVectorField_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> b,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
    for(int comp=0; comp<3; ++comp )
    {
      Db[iz][iy][ix][comp][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iz][iy][ix+1][comp] - b[iz][iy][ix-1][comp])/hX
                              : 
                              0.5*(b[iz][iy][ix][comp] - b[iz][iy][ix-1][comp])/hX
                            : 
                            0.5*(b[iz][iy][ix+1][comp] - b[iz][iy][ix][comp])/hX;

      Db[iz][iy][ix][comp][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iz][iy+1][ix][comp] - b[iz][iy-1][ix][comp])/hY
                              : 
                              0.5*(b[iz][iy][ix][comp] - b[iz][iy-1][ix][comp])/hY
                            : 
                            0.5*(b[iz][iy+1][ix][comp] - b[iz][iy][ix][comp])/hY;

      Db[iz][iy][ix][comp][2] = (iz > 0) ? 
                            (iz < NZ-1) ? 
                              0.5*(b[iz+1][iy][ix][comp] - b[iz-1][iy][ix][comp])/hZ
                              : 
                              0.5*(b[iz][iy][ix][comp] - b[iz-1][iy][ix][comp])/hZ
                            : 
                            0.5*(b[iz+1][iy][ix][comp] - b[iz][iy][ix][comp])/hZ;
    }
  }
}

template <typename T>
__global__ void cuda_nabla3d_cd_forwardVectorField_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> b,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
    for(int comp=0; comp<3; ++comp )
    {
      Db[iz][iy][ix][comp][0] = (ix > 0) ? 
                                  (ix < NX-1) ? 
                                    0.5*(b[iz][iy][ix+1][comp] - b[iz][iy][ix-1][comp])/hX
                                    : 
                                    0.
                                  : 
                                  0.;

      Db[iz][iy][ix][comp][1] = (iy > 0) ? 
                                  (iy < NY-1) ? 
                                    0.5*(b[iz][iy+1][ix][comp] - b[iz][iy-1][ix][comp])/hY
                                    : 
                                    0.
                                  : 
                                  0.;

      Db[iz][iy][ix][comp][2] = (iz > 0) ? 
                                (iz < NZ-1) ? 
                                  0.5*(b[iz+1][iy][ix][comp] - b[iz-1][iy][ix][comp])/hZ
                                  : 
                                  0.
                                : 
                                0.;
    }
  }
}

template <typename T>
__global__ void cuda_nabla3d_cd_forwardVectorField_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> b,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> Db)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ)
  {
    for(int comp=0; comp<3; ++comp )
    {
      Db[iz][iy][ix][comp][0] = (ix > 0) ? 
                            (ix < NX-1) ? 
                              0.5*(b[iz][iy][ix+1][comp] - b[iz][iy][ix-1][comp])/hX
                              : 
                              0.5*(b[iz][iy][ix][comp] - b[iz][iy][ix-1][comp])/hX
                            : 
                            0.5*(b[iz][iy][ix+1][comp] - b[iz][iy][ix][comp])/hX;

      Db[iz][iy][ix][comp][1] = (iy > 0) ? 
                            (iy < NY-1) ? 
                              0.5*(b[iz][iy+1][ix][comp] - b[iz][iy-1][ix][comp])/hY
                              : 
                              0.5*(b[iz][iy][ix][comp] - b[iz][iy-1][ix][comp])/hY
                            : 
                            0.5*(b[iz][iy+1][ix][comp] - b[iz][iy][ix][comp])/hY;

      Db[iz][iy][ix][comp][2] = (iz > 0) ? 
                            (iz < NZ-1) ? 
                              0.5*(b[iz+1][iy][ix][comp] - b[iz-1][iy][ix][comp])/hZ
                              : 
                              0.5*(b[iz][iy][ix][comp] - b[iz-1][iy][ix][comp])/hZ
                            : 
                            0.5*(b[iz+1][iy][ix][comp] - b[iz][iy][ix][comp])/hZ;
    }
  }
}


template <typename T>
__global__ void cuda_divergence3d_cd_backwardVectorField_bdryNearest_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> p,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    for(int comp=0; comp<3; ++comp )
    {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               0.5*(p[iz][iy][ix-1][comp][0] - p[iz][iy][ix+1][comp][0]) 
                               : 
                               0.5*(p[iz][iy][ix-1][comp][0] + p[iz][iy][ix][comp][0]) 
                              :
                              0.5*(-p[iz][iy][ix][comp][0] - p[iz][iy][ix+1][comp][0]);

      T divp_y = (iy > 0) ? 
                            (iy < NY - 1 ) ? 
                               0.5*(p[iz][iy-1][ix][comp][1] - p[iz][iy+1][ix][comp][1]) 
                               : 
                               0.5*(p[iz][iy-1][ix][comp][1] + p[iz][iy][ix][comp][1]) 
                              :
                              0.5*(-p[iz][iy][ix][comp][1] - p[iz][iy+1][ix][comp][1]);

      T divp_z = (iz > 0) ? 
                            (iz < NZ - 1 ) ? 
                               0.5*(p[iz-1][iy][ix][comp][2] - p[iz+1][iy][ix][comp][2]) 
                               : 
                               0.5*(p[iz-1][iy][ix][comp][2] + p[iz][iy][ix][comp][2]) 
                              :
                              0.5*(-p[iz][iy][ix][comp][2] - p[iz+1][iy][ix][comp][2]);

      divp[iz][iy][ix][comp] = divp_x/hX + divp_y/hY + divp_z/hZ;
    }
  }
}

template <typename T>
__global__ void cuda_divergence3d_cd_backwardVectorField_bdryMirror_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> p,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    for(int comp=0; comp<3; ++comp )
    {
      T divp_x = (ix > 1) ? 
                            (ix < NX - 2 ) ? 
                               0.5*(p[iz][iy][ix-1][comp][0] - p[iz][iy][ix+1][comp][0]) 
                               : 
                               0.5*p[iz][iy][ix-1][comp][0]
                              :
                              -0.5* p[iz][iy][ix+1][comp][0];

      T divp_y = (iy > 1) ? 
                            (iy < NY - 2 ) ? 
                               0.5*(p[iz][iy-1][ix][comp][1] - p[iz][iy+1][ix][comp][1]) 
                               : 
                               0.5*p[iz][iy-1][ix][comp][1]
                              :
                              -0.5*p[iz][iy+1][ix][comp][1];

      T divp_z = (iz > 1) ? 
                            (iz < NZ - 2 ) ? 
                               0.5*(p[iz-1][iy][ix][comp][2] - p[iz+1][iy][ix][comp][2]) 
                               : 
                               0.5*p[iz-1][iy][ix][comp][2]
                              :
                              -0.5*p[iz+1][iy][ix][comp][2];

      divp[iz][iy][ix][comp] = divp_x/hX + divp_y/hY + divp_z/hZ;
    }
  }
}


template <typename T>
__global__ void cuda_divergence3d_cd_backwardVectorField_bdryReflect_kernel(
  const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> p,
  const int NZ, const int NY, const int NX,
  const float hZ, const float hY, const float hX,
  torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> divp)
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int iz = blockDim.z * blockIdx.z + threadIdx.z;

  if (ix < NX && iy < NY && iz < NZ )
  {
    for(int comp=0; comp<3; ++comp )
    {
      T divp_x = (ix > 0) ? 
                            (ix < NX - 1 ) ? 
                               0.5*(p[iz][iy][ix-1][comp][0] - p[iz][iy][ix+1][comp][0]) 
                               : 
                               0.5*(p[iz][iy][ix-1][comp][0] + p[iz][iy][ix][comp][0]) 
                              :
                              0.5*(-p[iz][iy][ix][comp][0] - p[iz][iy][ix+1][comp][0]);

      T divp_y = (iy > 0) ? 
                            (iy < NY - 1 ) ? 
                               0.5*(p[iz][iy-1][ix][comp][1] - p[iz][iy+1][ix][comp][1]) 
                               : 
                               0.5*(p[iz][iy-1][ix][comp][1] + p[iz][iy][ix][comp][1]) 
                              :
                              0.5*(-p[iz][iy][ix][comp][1] - p[iz][iy+1][ix][comp][1]);

      T divp_z = (iz > 0) ? 
                            (iz < NZ - 1 ) ? 
                               0.5*(p[iz-1][iy][ix][comp][2] - p[iz+1][iy][ix][comp][2]) 
                               : 
                               0.5*(p[iz-1][iy][ix][comp][2] + p[iz][iy][ix][comp][2]) 
                              :
                              0.5*(-p[iz][iy][ix][comp][2] - p[iz+1][iy][ix][comp][2]);

      divp[iz][iy][ix][comp] = divp_x/hX + divp_y/hY + divp_z/hZ;
    }
  }
}


//=========================================================
// C++ kernel calls
//=========================================================

torch::Tensor cuda_nabla1d_cd_forward(
  const torch::Tensor &b, const MeshInfo1D &meshInfo, const BoundaryType boundary )
{
  TORCH_CHECK(b.dim() == 1, "Expected 1d tensor");

  const int NX = b.size(0);
  const float hX = meshInfo.gethX();

  auto Db = torch::zeros({NX, 1}, b.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(boundary)
  {

    case BOUNDARY_NEAREST:
    AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla1d_cd_forward", ([&]{
      cuda_nabla1d_cd_forward_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
        b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        NX,
        hX,
        Db.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;


    case BOUNDARY_MIRROR:
    AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla1d_cd_forward", ([&]{
      cuda_nabla1d_cd_forward_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
        b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        NX,
        hX,
        Db.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_REFLECT:
    AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla1d_cd_forward", ([&]{
      cuda_nabla1d_cd_forward_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
        b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        NX,
        hX,
        Db.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
    cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}

torch::Tensor cuda_divergence1d_cd_backward(
  const torch::Tensor &p, const MeshInfo1D &meshInfo, const BoundaryType boundary)
{
  TORCH_CHECK(p.dim() == 2, "Expected 2d tensor");

  const int NX = p.size(0);
  const float hX = meshInfo.gethX();

  auto divp = torch::zeros({NX}, p.options());

  const dim3 blockSize(512, 1, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

switch(boundary)
  {

      case BOUNDARY_NEAREST:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence1d_cd_backward", ([&]{
        cuda_divergence1d_cd_backward_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          NX, 
          hX,
          divp.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
      break;

      case BOUNDARY_MIRROR:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence1d_cd_backward", ([&]{
        cuda_divergence1d_cd_backward_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          NX, 
          hX,
          divp.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
      break;

      case BOUNDARY_REFLECT:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence1d_cd_backward", ([&]{
        cuda_divergence1d_cd_backward_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          NX, 
          hX,
          divp.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
      break;
  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}


torch::Tensor cuda_nabla2d_cd_forward(
  const torch::Tensor &b, const MeshInfo2D &meshInfo, const BoundaryType boundary)
{
  TORCH_CHECK(b.dim() == 2, "Expected 2d tensor");

  const int NY = b.size(0);
  const int NX = b.size(1);
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto Db = torch::zeros({NY, NX, 2}, b.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                      (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(boundary)
  {

      case BOUNDARY_NEAREST:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla2d_cd_forward", ([&]{
        cuda_nabla2d_cd_forward_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          Db.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
     break;

      case BOUNDARY_MIRROR:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla2d_cd_forward", ([&]{
        cuda_nabla2d_cd_forward_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          Db.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
     break;

      case BOUNDARY_REFLECT:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla2d_cd_forward", ([&]{
        cuda_nabla2d_cd_forward_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          Db.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
     break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}


torch::Tensor cuda_divergence2d_cd_backward(
  const torch::Tensor &p, const MeshInfo2D &meshInfo, const BoundaryType boundary)
{
  TORCH_CHECK(p.dim() == 3, "Expected 3d tensor");

  const int NY = p.size(0);
  const int NX = p.size(1);
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto divp = torch::zeros({NY,NX}, p.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(boundary)
  {

    case BOUNDARY_NEAREST:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence2d_cd_backward", ([&]{
        cuda_divergence2d_cd_backward_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          divp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;


    case BOUNDARY_MIRROR:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence2d_cd_backward", ([&]{
        cuda_divergence2d_cd_backward_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          divp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;


    case BOUNDARY_REFLECT:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence2d_cd_backward", ([&]{
        cuda_divergence2d_cd_backward_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          divp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}


torch::Tensor cuda_nabla2d_cd_forwardVectorField(
  const torch::Tensor &b, const MeshInfo2D &meshInfo, const BoundaryType boundary)
{
  TORCH_CHECK(b.dim() == 3, "Expected 3d tensor");

  const int NY = b.size(0);
  const int NX = b.size(1);
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto Db = torch::zeros({NY, NX, 2, 2}, b.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(boundary)
  {

    case BOUNDARY_NEAREST:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla2d_cd_forwardVectorField", ([&]{
        cuda_nabla2d_cd_forwardVectorField_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          Db.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_MIRROR:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla2d_cd_forwardVectorField", ([&]{
        cuda_nabla2d_cd_forwardVectorField_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          Db.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_REFLECT:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla2d_cd_forwardVectorField", ([&]{
        cuda_nabla2d_cd_forwardVectorField_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          Db.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;
  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}


torch::Tensor cuda_divergence2d_cd_backwardVectorField(
  const torch::Tensor &p, const MeshInfo2D &meshInfo, const BoundaryType boundary)
{
  TORCH_CHECK(p.dim() == 4, "Expected 4d tensor");

  const int NY = p.size(0);
  const int NX = p.size(1);
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto divp = torch::zeros({NY,NX,2}, p.options());

  const dim3 blockSize(32, 32, 1); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(boundary)
  {

    case BOUNDARY_NEAREST:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence2d_cd_backwardVectorField", ([&]{
        cuda_divergence2d_cd_backwardVectorField_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          divp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_MIRROR:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence2d_cd_backwardVectorField", ([&]{
        cuda_divergence2d_cd_backwardVectorField_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          divp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_REFLECT:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence2d_cd_backwardVectorField", ([&]{
        cuda_divergence2d_cd_backwardVectorField_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          NY, NX,
          hY, hX,
          divp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}



torch::Tensor cuda_nabla3d_cd_forward(
  const torch::Tensor &b, const MeshInfo3D &meshInfo, const BoundaryType boundary)
{
  TORCH_CHECK(b.dim() == 3, "Expected 3d tensor");

  const int NZ = b.size(0);
  const int NY = b.size(1);
  const int NX = b.size(2);
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto Db = torch::zeros({NZ,NY,NX, 3}, b.options());

  const dim3 blockSize(16,16,3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                      (NY + blockSize.y - 1) / blockSize.y,
                      (NZ + blockSize.z - 1) / blockSize.z);

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(boundary)
  {

    case BOUNDARY_NEAREST:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla3d_cd_forward", ([&]{
        cuda_nabla3d_cd_forward_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          Db.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_MIRROR:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla3d_cd_forward", ([&]{
        cuda_nabla3d_cd_forward_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          Db.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_REFLECT:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla3d_cd_forward", ([&]{
        cuda_nabla3d_cd_forward_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          Db.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}

torch::Tensor cuda_divergence3d_cd_backward(
  const torch::Tensor &p, const MeshInfo3D &meshInfo, const BoundaryType boundary)
{
  TORCH_CHECK(p.dim() == 4, "Expected 4d tensor");

  const int NZ = p.size(0);
  const int NY = p.size(1);
  const int NX = p.size(2);
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto divp = torch::zeros({NZ,NY,NX}, p.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y,
                       (NZ + blockSize.z - 1) / blockSize.z  );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif


  switch(boundary)
  {

    case BOUNDARY_NEAREST:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence3d_cd_backward", ([&]{
        cuda_divergence3d_cd_backward_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          divp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_MIRROR:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence3d_cd_backward", ([&]{
        cuda_divergence3d_cd_backward_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          divp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_REFLECT:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence3d_cd_backward", ([&]{
        cuda_divergence3d_cd_backward_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          divp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}

torch::Tensor cuda_nabla3d_cd_forwardVectorField(
  const torch::Tensor &b, const MeshInfo3D &meshInfo, const BoundaryType boundary)
{
  TORCH_CHECK(b.dim() == 4, "Expected 4d tensor");

  const int NZ = b.size(0);
  const int NY = b.size(1);
  const int NX = b.size(2);
  //const int numComp = b.size(3); //should be 3
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto Db = torch::zeros({NZ,NY,NX,3,3}, b.options());

  const dim3 blockSize(16,16,3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                      (NY + blockSize.y - 1) / blockSize.y,
                      (NZ + blockSize.z - 1) / blockSize.z);

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(boundary)
  {

    case BOUNDARY_NEAREST:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla3d_cd_forwardVectorField", ([&]{
        cuda_nabla3d_cd_forwardVectorField_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          Db.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_MIRROR:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla3d_cd_forwardVectorField", ([&]{
        cuda_nabla3d_cd_forwardVectorField_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          Db.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_REFLECT:
      AT_DISPATCH_FLOATING_TYPES(b.type(), "nabla3d_cd_forwardVectorField", ([&]{
        cuda_nabla3d_cd_forwardVectorField_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
          b.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          Db.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return Db;
}

torch::Tensor cuda_divergence3d_cd_backwardVectorField(
  const torch::Tensor &p, const MeshInfo3D &meshInfo, const BoundaryType boundary)
{
  TORCH_CHECK(p.dim() == 5, "Expected 5d tensor");

  const int NZ = p.size(0);
  const int NY = p.size(1);
  const int NX = p.size(2);
  const float hZ = meshInfo.gethZ();
  const float hY = meshInfo.gethY();
  const float hX = meshInfo.gethX();

  auto divp = torch::zeros({NZ,NY,NX,3}, p.options());

  const dim3 blockSize(16, 16, 3); 
  const dim3 numBlocks((NX + blockSize.x - 1) / blockSize.x,
                       (NY + blockSize.y - 1) / blockSize.y,
                       (NZ + blockSize.z - 1) / blockSize.z  );

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  switch(boundary)
  {

    case BOUNDARY_NEAREST:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence3d_cd_backwardVectorField", ([&]{
        cuda_divergence3d_cd_backwardVectorField_bdryNearest_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          divp.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_MIRROR:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence3d_cd_backwardVectorField", ([&]{
        cuda_divergence3d_cd_backwardVectorField_bdryMirror_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          divp.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

    case BOUNDARY_REFLECT:
      AT_DISPATCH_FLOATING_TYPES(p.type(), "divergence3d_cd_backwardVectorField", ([&]{
        cuda_divergence3d_cd_backwardVectorField_bdryReflect_kernel<scalar_t><<<numBlocks, blockSize>>>(
          p.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          NZ, NY, NX,
          hZ, hY, hX,
          divp.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
      }));
      cudaSafeCall(cudaGetLastError());
    break;

  }

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << std::endl;
#endif

  return divp;
}

