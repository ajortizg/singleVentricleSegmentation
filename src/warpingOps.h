#ifndef __WARPINGOPS_H_
#define __WARPINGOPS_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"
#include "boundary.cuh"

//=======================================
// CUDA forward declarations
//=======================================

torch::Tensor cuda_warp1d( const torch::Tensor &u, const torch::Tensor &phi, const MeshInfo1D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary );
std::vector<torch::Tensor> cuda_warp1d_backward( const torch::Tensor u, const torch::Tensor phi, const torch::Tensor forward_out, const MeshInfo1D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary);

torch::Tensor cuda_warp2d( const torch::Tensor &u, const torch::Tensor &phi, const MeshInfo2D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary );
std::vector<torch::Tensor> cuda_warp2d_backward( const torch::Tensor u, const torch::Tensor phi, const torch::Tensor forward_out, const MeshInfo2D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary);
torch::Tensor cuda_warpVectorField2d( const torch::Tensor &u, const torch::Tensor &phi, const MeshInfo2D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary );

torch::Tensor cuda_warp3d( const torch::Tensor &u, const torch::Tensor &phi, const MeshInfo3D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary );
std::vector<torch::Tensor> cuda_warp3d_backward( const torch::Tensor u, const torch::Tensor phi, const torch::Tensor forward_out, const MeshInfo3D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary);
torch::Tensor cuda_warpVectorField3d( const torch::Tensor &u, const torch::Tensor &phi, const MeshInfo3D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary );


//=======================================
// C++ interface
//=======================================

class Warping1D {
public:

  const MeshInfo1D & _meshInfo;
  const InterpolationType _interpolation;
  const BoundaryType _boundary;

  Warping1D( const MeshInfo1D & meshInfo, const InterpolationType interpolation, const BoundaryType boundary ) 
  : _meshInfo(meshInfo), _interpolation ( interpolation ), _boundary( boundary ) {}

  torch::Tensor forward(const torch::Tensor &u, const torch::Tensor &phi) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    return cuda_warp1d(u,phi,_meshInfo,_interpolation,_boundary);
  }

  std::vector<torch::Tensor> backward(const torch::Tensor u, const torch::Tensor phi, const torch::Tensor forward_out) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    CHECK_INPUT(forward_out);
    return cuda_warp1d_backward(u,phi,forward_out,_meshInfo,_interpolation,_boundary);
  }

};



class Warping2D {
public:

  const MeshInfo2D & _meshInfo;
  const InterpolationType _interpolation;
  const BoundaryType _boundary;

  Warping2D( const MeshInfo2D & meshInfo, const InterpolationType interpolation, const BoundaryType boundary ) : 
  _meshInfo(meshInfo), _interpolation ( interpolation ), _boundary( boundary ) {}

  torch::Tensor forward(const torch::Tensor &u, const torch::Tensor &phi ) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    return cuda_warp2d(u,phi,_meshInfo,_interpolation,_boundary);
  }

  std::vector<torch::Tensor> backward(const torch::Tensor u, const torch::Tensor phi, const torch::Tensor forward_out) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    CHECK_INPUT(forward_out);
    return cuda_warp2d_backward(u,phi,forward_out,_meshInfo,_interpolation,_boundary);
  }

  torch::Tensor forwardVectorField(const torch::Tensor &u, const torch::Tensor &phi ) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    return cuda_warpVectorField2d(u,phi,_meshInfo,_interpolation,_boundary);
  }

};

class Warping3D {
public:

  const MeshInfo3D & _meshInfo;
  const InterpolationType _interpolation;
  const BoundaryType _boundary;

  Warping3D( const MeshInfo3D & meshInfo, const InterpolationType interpolation, const BoundaryType boundary ) 
  : _meshInfo(meshInfo), _interpolation ( interpolation ), _boundary( boundary ) {}

  torch::Tensor forward(const torch::Tensor &u, const torch::Tensor &phi ) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    return cuda_warp3d(u,phi,_meshInfo,_interpolation,_boundary);
  }

  std::vector<torch::Tensor> backward(const torch::Tensor u, const torch::Tensor phi, const torch::Tensor forward_out) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    CHECK_INPUT(forward_out);
    return cuda_warp3d_backward(u,phi,forward_out,_meshInfo,_interpolation,_boundary);
  }

  torch::Tensor forwardVectorField(const torch::Tensor &u, const torch::Tensor &phi ) const {
    CHECK_INPUT(u);
    CHECK_INPUT(phi);
    return cuda_warpVectorField3d(u,phi,_meshInfo,_interpolation,_boundary);
  }

};

#endif