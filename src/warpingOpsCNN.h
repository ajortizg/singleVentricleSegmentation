#ifndef __WARPINGOPSCNN_H_
#define __WARPINGOPSCNN_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"
#include "boundary.cuh"

//=======================================
// CUDA forward declarations
//=======================================
torch::Tensor cuda_warpCNN3d( const torch::Tensor &u, const torch::Tensor &phi, const MeshInfo3D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary );
// std::vector<torch::Tensor> cuda_warp3d_backward( const torch::Tensor u, const torch::Tensor phi, const torch::Tensor forward_out, const MeshInfo3D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary);


//=======================================
// C++ interface
//=======================================

class WarpingCNN3D {
public:

  const MeshInfo3D & _meshInfo;
  const InterpolationType _interpolation;
  const BoundaryType _boundary;

  WarpingCNN3D( const MeshInfo3D & meshInfo, const InterpolationType interpolation, const BoundaryType boundary ) 
  : _meshInfo(meshInfo), _interpolation ( interpolation ), _boundary( boundary ) {}

  torch::Tensor forward(const torch::Tensor &u, const torch::Tensor &phi ) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    return cuda_warpCNN3d(u,phi,_meshInfo,_interpolation,_boundary);
  }

//   std::vector<torch::Tensor> backward(const torch::Tensor u, const torch::Tensor phi, const torch::Tensor forward_out) const {
//     CHECK_INPUT(u);
//     CHECK_INPUT(phi);
//     CHECK_INPUT(forward_out);
//     return cuda_warpCNN3d_backward(u,phi,forward_out,_meshInfo,_interpolation,_boundary);
//   }

};

#endif