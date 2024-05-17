#ifndef __INTERPOLATION_CUH_
#define __INTERPOLATION_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h> 
#include <iostream>
#include <stdio.h>

#include "interpolation_cubicHermiteSpline.cuh"
#include "interpolation_linear.cuh"
#include "interpolation_nearest.cuh"

// typedef float (*get_interpolate_func) (const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits>, int, float, float, int, float);

// __device__ get_interpolate_func interpolationFunc[3] = { cuda_interpolate1d_nearest, cuda_interpolate1d_linear, cuda_interpolate1d_cubicHermiteSpline };

// __device__ float cuda_interpolation_1d(
//        const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> u, 
//        const int NX, const float LX, const float hX,
//        const int interpolation,
//        const int boundary,
//        const float coord_x_warped) {
//   return interpolationFunc[interpolation](u,NX,LX,hX,boundary,coord_x_warped);
// }

#endif
