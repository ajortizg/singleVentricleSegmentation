#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 

#include <iostream>
#include <stdio.h>

#include "coreDefines.h"


//NEAREST Boundary
inline __device__ int getIndex_interpolate_bdryNearest( int ix_in, int N) 
{
  if( ix_in < 0 ) return 0;
  else if( ix_in > N - 1 ) return N-1;
  else return ix_in;
}

//MIRROR Boundary
inline __device__ int getIndex_interpolate_bdryMirror( int ix_in, int N) 
{
  if( ix_in < 0 ) return -ix_in - 1;
  else if( ix_in > N - 1 ) return 2*N-1-ix_in;
  else return ix_in;
}

//REFLECTION Boundary
inline __device__ int getIndex_interpolate_bdryReflect( int ix_in, int N) 
{
  if( ix_in < 0 ) return -ix_in;
  else if( ix_in > N - 1 ) return 2*N-2-ix_in;
  else return ix_in;
}



//=============================
// (multi-)linear interpolation

//ZERO Boundary
template <typename T>
__device__ T cuda_interpolate1d_linear(
       const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, 
       const int NX, const float LX, const float hX,
       const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX);
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

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

//REFLECTION Boundary
template <typename T>
__device__ T cuda_interpolate1d_linear_bdryReflect(
      const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> u, 
      const int NX, const float LX, const float hX,
      const T coord_x_warped) 
{
 const int ix_f = floorf(coord_x_warped / hX);
 const int ix_c = ix_f + 1;
 const T wx = coord_x_warped / hX - ix_f;
 const int ix_f_out = getIndex_interpolate_bdryReflect(ix_f,NX);  
 const int ix_c_out = getIndex_interpolate_bdryReflect(ix_c,NX);  

 T u_f = u[ix_f_out];
 T u_c = u[ix_c_out];

 T out = (1 - wx) * u_f;
 out += wx * u_c;

 return out;
}


//ZERO Boundary
template <typename T>
__device__ T cuda_interpolate2d_bilinear(const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

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


//REFLECTION Boundary
template <typename T>
__device__ T cuda_interpolate2d_bilinear_bdryReflect(const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndex_interpolate_bdryReflect(ix_f,NX);
  const int ix_c_out = getIndex_interpolate_bdryReflect(ix_c,NX);

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;
  const int iy_f_out = getIndex_interpolate_bdryReflect(iy_f,NY);
  const int iy_c_out = getIndex_interpolate_bdryReflect(iy_c,NY);

  T u_ff = u[iy_f_out][ix_f_out];
  T u_fc = u[iy_c_out][ix_f_out];
  T u_cf = u[iy_f_out][ix_c_out];
  T u_cc = u[iy_c_out][ix_c_out];

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}





template <typename T>
__device__ T cuda_interpolateVectorField2d_bilinear(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const T coord_y_warped, const T coord_x_warped,
                                         const int comp) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  T u_ff = 0, u_fc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_ff = u[iy_f][ix_f][comp];

    if (iy_c >= 0 && iy_c < NY)
      u_fc = u[iy_c][ix_f][comp];
  }

  T u_cf = 0, u_cc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_cf = u[iy_f][ix_c][comp];

    if (iy_c >= 0 && iy_c < NY)
      u_cc = u[iy_c][ix_c][comp];
  }

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}


template <typename T>
__device__ T cuda_interpolateMatrixField2d_bilinear(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
                                         const int NY, const int NX,
                                         const float LY, const float LX,
                                         const float hY, const float hX,
                                         const T coord_y_warped, const T coord_x_warped,
                                         const int comp_i, const int comp_j ) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  T u_ff = 0, u_fc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_ff = u[iy_f][ix_f][comp_i][comp_j];

    if (iy_c >= 0 && iy_c < NY)
      u_fc = u[iy_c][ix_f][comp_i][comp_j];
  }

  T u_cf = 0, u_cc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY)
      u_cf = u[iy_f][ix_c][comp_i][comp_j];

    if (iy_c >= 0 && iy_c < NY)
      u_cc = u[iy_c][ix_c][comp_i][comp_j];
  }

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  return out;
}


//ZERO Boundary
template <typename T>
__device__ T cuda_interpolate3d_trilinear(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;

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
  out += wz * wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wz * wy * wx * u_ccc;

  return out;
}




//REFLECTION Boundary
template <typename T>
__device__ T cuda_interpolate3d_trilinear_bdryReflection(const torch::PackedTensorAccessor32<T,3,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndex_interpolate_bdryReflect(ix_f,NX);
  const int ix_c_out = getIndex_interpolate_bdryReflect(ix_c,NX);

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;
  const int iy_f_out = getIndex_interpolate_bdryReflect(iy_f,NY);
  const int iy_c_out = getIndex_interpolate_bdryReflect(iy_c,NY);

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;
  const int iz_f_out = getIndex_interpolate_bdryReflect(iz_f,NZ);
  const int iz_c_out = getIndex_interpolate_bdryReflect(iz_c,NZ);

  T u_fff = u[iz_f_out][iy_f_out][ix_f_out];
  T u_ffc = u[iz_c_out][iy_f_out][ix_f_out];
  T u_fcf = u[iz_f_out][iy_c_out][ix_f_out];
  T u_fcc = u[iz_c_out][iy_c_out][ix_f_out];
  T u_cff = u[iz_f_out][iy_f_out][ix_c_out];
  T u_cfc = u[iz_c_out][iy_f_out][ix_c_out];
  T u_ccf = u[iz_f_out][iy_c_out][ix_c_out];
  T u_ccc = u[iz_c_out][iy_c_out][ix_c_out];

  T out = (1 - wz) * (1 - wy) * (1 - wx) * u_fff;
  out += wz * (1 - wy) * (1 - wx) * u_ffc;
  out += (1 - wz) * (1 - wy) * wx * u_cff;
  out += wz * (1 - wy) * wx * u_cfc;
  out += (1 - wz) * wy * (1 - wx) * u_fcf;
  out += wz * wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wz * wy * wx * u_ccc;

  return out;
}


template <typename T>
__device__ T cuda_interpolateVectorField3d_trilinear(const torch::PackedTensorAccessor32<T,4,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped, 
                                          const int comp) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;

  T u_fff = 0, u_ffc = 0, u_fcf = 0, u_fcc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fff = u[iz_f][iy_f][ix_f][comp];

        if (iz_c >= 0 && iz_c < NZ)
          u_ffc = u[iz_c][iy_f][ix_f][comp];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fcf = u[iz_f][iy_c][ix_f][comp];

        if (iz_c >= 0 && iz_c < NZ)
          u_fcc = u[iz_c][iy_c][ix_f][comp];
    }
  }

  T u_cff = 0, u_cfc = 0, u_ccf = 0, u_ccc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_cff = u[iz_f][iy_f][ix_c][comp];

        if (iz_c >= 0 && iz_c < NZ)
          u_cfc = u[iz_c][iy_f][ix_c][comp];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_ccf = u[iz_f][iy_c][ix_c][comp];

        if (iz_c >= 0 && iz_c < NZ)
          u_ccc = u[iz_c][iy_c][ix_c][comp];
    }
  }

  T out = (1 - wz) * (1 - wy) * (1 - wx) * u_fff;
  out += wz * (1 - wy) * (1 - wx) * u_ffc;
  out += (1 - wz) * (1 - wy) * wx * u_cff;
  out += wz * (1 - wy) * wx * u_cfc;
  out += (1 - wz) * wy * (1 - wx) * u_fcf;
  out += wz * wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wz * wy * wx * u_ccc;

  return out;
}





template <typename T>
__device__ T cuda_interpolateMatrixField3d_trilinear(const torch::PackedTensorAccessor32<T,5,torch::RestrictPtrTraits> u,
                                          const int NZ, const int NY, const int NX,
                                          const float LZ, const float LY, const float LX,
                                          const float hZ, const float hY, const float hX,
                                          const T coord_z_warped, const T coord_y_warped, const T coord_x_warped, 
                                          const int comp_i, const int comp_j ) {
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;

  const int iz_f = floorf(coord_z_warped / hZ );
  const int iz_c = iz_f + 1;
  const T wz = coord_z_warped / hZ - iz_f;

  T u_fff = 0, u_ffc = 0, u_fcf = 0, u_fcc = 0;
  if (ix_f >= 0 && ix_f < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fff = u[iz_f][iy_f][ix_f][comp_i][comp_j];

        if (iz_c >= 0 && iz_c < NZ)
          u_ffc = u[iz_c][iy_f][ix_f][comp_i][comp_j];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_fcf = u[iz_f][iy_c][ix_f][comp_i][comp_j];

        if (iz_c >= 0 && iz_c < NZ)
          u_fcc = u[iz_c][iy_c][ix_f][comp_i][comp_j];
    }
  }

  T u_cff = 0, u_cfc = 0, u_ccf = 0, u_ccc = 0;
  if (ix_c >= 0 && ix_c < NX) {
    if (iy_f >= 0 && iy_f < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_cff = u[iz_f][iy_f][ix_c][comp_i][comp_j];

        if (iz_c >= 0 && iz_c < NZ)
          u_cfc = u[iz_c][iy_f][ix_c][comp_i][comp_j];
    }
      
    if (iy_c >= 0 && iy_c < NY){
        if (iz_f >= 0 && iz_f < NZ)
          u_ccf = u[iz_f][iy_c][ix_c][comp_i][comp_j];

        if (iz_c >= 0 && iz_c < NZ)
          u_ccc = u[iz_c][iy_c][ix_c][comp_i][comp_j];
    }
  }

  T out = (1 - wz) * (1 - wy) * (1 - wx) * u_fff;
  out += wz * (1 - wy) * (1 - wx) * u_ffc;
  out += (1 - wz) * (1 - wy) * wx * u_cff;
  out += wz * (1 - wy) * wx * u_cfc;
  out += (1 - wz) * wy * (1 - wx) * u_fcf;
  out += wz * wy * (1 - wx) * u_fcc;
  out += (1 - wz) * wy * wx * u_ccf;
  out += wz * wy * wx * u_ccc;

  return out;
}