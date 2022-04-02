#ifndef __DIFFERENTIALOPS_H_
#define __DIFFERENTIALOPS_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"
#include "boundary.cuh"

//=======================================
// CUDA forward declarations
//=======================================

torch::Tensor cuda_nabla1d_cd_forward( const torch::Tensor &b, const MeshInfo1D &meshInfo, const BoundaryType boundary);
torch::Tensor cuda_divergence1d_cd_backward( const torch::Tensor &b, const MeshInfo1D &meshInfo, const BoundaryType boundary);

torch::Tensor cuda_nabla2d_cd_forward( const torch::Tensor &b, const MeshInfo2D &meshInfo, const BoundaryType boundary);
torch::Tensor cuda_divergence2d_cd_backward( const torch::Tensor &b, const MeshInfo2D &meshInfo, const BoundaryType boundary);
torch::Tensor cuda_nabla2d_cd_forwardVectorField( const torch::Tensor &b, const MeshInfo2D &meshInfo, const BoundaryType boundary);
torch::Tensor cuda_divergence2d_cd_backwardVectorField( const torch::Tensor &b, const MeshInfo2D &meshInfo, const BoundaryType boundary);

torch::Tensor cuda_nabla3d_cd_forward( const torch::Tensor &b, const MeshInfo3D &meshInfo, const BoundaryType boundary);
torch::Tensor cuda_divergence3d_cd_backward( const torch::Tensor &b, const MeshInfo3D &meshInfo, const BoundaryType boundary);
torch::Tensor cuda_nabla3d_cd_forwardVectorField( const torch::Tensor &b, const MeshInfo3D &meshInfo, const BoundaryType boundary);
torch::Tensor cuda_divergence3d_cd_backwardVectorField( const torch::Tensor &b, const MeshInfo3D &meshInfo, const BoundaryType boundary);

//=======================================
// C++ interface
//=======================================

class Nabla1D_CD {
public:

  const MeshInfo1D & _meshInfo;
  const BoundaryType _boundary;

  Nabla1D_CD( const MeshInfo1D & meshInfo, const BoundaryType boundary ) : _meshInfo(meshInfo), _boundary(boundary) {}

  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla1d_cd_forward(b,_meshInfo,_boundary);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence1d_cd_backward(b,_meshInfo,_boundary);
  }
};

class Nabla2D_CD {
public:

  const MeshInfo2D & _meshInfo;
  const BoundaryType _boundary;

  Nabla2D_CD( const MeshInfo2D & meshInfo, const BoundaryType boundary ) : _meshInfo(meshInfo), _boundary(boundary) {}

  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla2d_cd_forward(b,_meshInfo,_boundary);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence2d_cd_backward(b,_meshInfo,_boundary);
  }
  torch::Tensor forwardVectorField(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla2d_cd_forwardVectorField(b,_meshInfo,_boundary);
  }
  torch::Tensor backwardVectorField(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence2d_cd_backwardVectorField(b,_meshInfo,_boundary);
  }
};

class Nabla3D_CD {
public:

  const MeshInfo3D & _meshInfo;
  const BoundaryType _boundary;

  Nabla3D_CD( const MeshInfo3D & meshInfo, const BoundaryType boundary ) : _meshInfo(meshInfo), _boundary(boundary) {}
  
  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla3d_cd_forward(b,_meshInfo,_boundary);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence3d_cd_backward(b,_meshInfo,_boundary);
  }
  torch::Tensor forwardVectorField(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla3d_cd_forwardVectorField(b,_meshInfo,_boundary);
  }
  torch::Tensor backwardVectorField(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence3d_cd_backwardVectorField(b,_meshInfo,_boundary);
  }
};


#endif