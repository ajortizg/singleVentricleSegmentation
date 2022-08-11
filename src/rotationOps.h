#ifndef __ROTATIONOPS_H_
#define __ROTATIONOPS_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"
#include "boundary.cuh"

//=======================================
// CUDA forward declarations
//=======================================
torch::Tensor cuda_rotate3d( const torch::Tensor &u, const torch::Tensor &rotationMat, const torch::Tensor &offset, const MeshInfo3D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary );
torch::Tensor cuda_rotateVectorField3d( const torch::Tensor &u,  const torch::Tensor &rotationMat, const torch::Tensor &offset, const MeshInfo3D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary );
torch::Tensor cuda_rotateMatrixField3d( const torch::Tensor &u,  const torch::Tensor &rotationMat, const torch::Tensor &offset, const MeshInfo3D& meshInfo, const InterpolationType interpolation, const BoundaryType boundary );


//=======================================
// C++ interface
//=======================================

class Rotation3D {
public:

  const MeshInfo3D & _meshInfo;
  const InterpolationType _interpolation;
  const BoundaryType _boundary;

  Rotation3D( const MeshInfo3D & meshInfo, const InterpolationType interpolation, const BoundaryType boundary ) 
  : _meshInfo(meshInfo), _interpolation ( interpolation ), _boundary( boundary ) {}

  torch::Tensor forward(const torch::Tensor &u, const torch::Tensor &rotationMat, const torch::Tensor &offset ) const {
    CHECK_INPUT(u);
    CHECK_INPUT(rotationMat);
    CHECK_INPUT(offset);
    return cuda_rotate3d(u,rotationMat,offset,_meshInfo,_interpolation,_boundary);
  }

  torch::Tensor forwardVectorField(const torch::Tensor &u, const torch::Tensor &rotationMat, const torch::Tensor &offset ) const {
    CHECK_INPUT(u);
    CHECK_INPUT(rotationMat);
    CHECK_INPUT(offset);
    return cuda_rotateVectorField3d(u,rotationMat,offset,_meshInfo,_interpolation,_boundary);
  }

  torch::Tensor forwardMatrixField(const torch::Tensor &u, const torch::Tensor &rotationMat, const torch::Tensor &offset ) const {
    CHECK_INPUT(u);
    CHECK_INPUT(rotationMat);
    CHECK_INPUT(offset);
    return cuda_rotateMatrixField3d(u,rotationMat,offset,_meshInfo,_interpolation,_boundary);
  }

};

#endif