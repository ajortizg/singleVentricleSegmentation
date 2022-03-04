#ifndef __WARPINGOPS_H_
#define __WARPINGOPS_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"

//=======================================
// CUDA forward declarations
//=======================================

torch::Tensor cuda_warp1d( const torch::Tensor &u, const torch::Tensor &phi, const MeshInfo1D& meshInfo, const InterpolationType interpolation = INTERPOLATE_LINEAR);
torch::Tensor cuda_warp2d( const torch::Tensor &u, const torch::Tensor &phi, const MeshInfo2D& meshInfo, const InterpolationType interpolation = INTERPOLATE_LINEAR);
torch::Tensor cuda_warp3d( const torch::Tensor &u, const torch::Tensor &phi, const MeshInfo3D& meshInfo, const InterpolationType interpolation = INTERPOLATE_LINEAR);

//=======================================
// C++ interface
//=======================================

class Warping1D {
public:

  const MeshInfo1D & _meshInfo;

  Warping1D( const MeshInfo1D & meshInfo ) : _meshInfo(meshInfo) {}

  torch::Tensor forward(const torch::Tensor &u, const torch::Tensor &phi, const InterpolationType interpolation = INTERPOLATE_LINEAR) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    return cuda_warp1d(u,phi,_meshInfo,interpolation);
  }
  // torch::Tensor backward(const torch::Tensor &b) const{
  //   CHECK_INPUT(b);
  // }
};



class Warping2D {
public:

  const MeshInfo2D & _meshInfo;

  Warping2D( const MeshInfo2D & meshInfo ) : _meshInfo(meshInfo) {}

  torch::Tensor forward(const torch::Tensor &u, const torch::Tensor &phi, const InterpolationType interpolation = INTERPOLATE_LINEAR) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    return cuda_warp2d(u,phi,_meshInfo,interpolation);
  }
  // torch::Tensor backward(const torch::Tensor &b) const{
  //   CHECK_INPUT(b);
  // }
};

class Warping3D {
public:

  const MeshInfo3D & _meshInfo;

  Warping3D( const MeshInfo3D & meshInfo ) : _meshInfo(meshInfo) {}

  torch::Tensor forward(const torch::Tensor &u, const torch::Tensor &phi, const InterpolationType interpolation = INTERPOLATE_LINEAR) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    return cuda_warp3d(u,phi,_meshInfo,interpolation);
  }
  // torch::Tensor backward(const torch::Tensor &b) const{
  //   CHECK_INPUT(b);
  // }
};

#endif