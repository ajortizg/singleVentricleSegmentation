#ifndef __PROLONGATIONOPS_H_
#define __PROLONGATIONOPS_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"
#include "boundary.cuh"

//=======================================
// CUDA forward declarations
//=======================================

torch::Tensor cuda_prolongate1d( const torch::Tensor &u, const MeshInfo1D& meshInfo, const MeshInfo1D& meshInfoProlongated, const InterpolationType interpolation, const BoundaryType boundary);

torch::Tensor cuda_prolongate2d( const torch::Tensor &u, const MeshInfo2D& meshInfo, const MeshInfo2D& meshInfoProlongated, const InterpolationType interpolation, const BoundaryType boundary);
torch::Tensor cuda_prolongateVectorField2d( const torch::Tensor &u, const MeshInfo2D& meshInfo, const MeshInfo2D& meshInfoProlongated, const InterpolationType interpolation, const BoundaryType boundary);
torch::Tensor cuda_prolongateMatrixField2d( const torch::Tensor &u, const MeshInfo2D& meshInfo, const MeshInfo2D& meshInfoProlongated, const InterpolationType interpolation, const BoundaryType boundary);

torch::Tensor cuda_prolongate3d( const torch::Tensor &u, const MeshInfo3D& meshInfo, const MeshInfo3D& meshInfoProlongated, const InterpolationType interpolation, const BoundaryType boundary);
torch::Tensor cuda_prolongateVectorField3d( const torch::Tensor &u, const MeshInfo3D& meshInfo, const MeshInfo3D& meshInfoProlongated, const InterpolationType interpolation, const BoundaryType boundary);
torch::Tensor cuda_prolongateMatrixField3d( const torch::Tensor &u, const MeshInfo3D& meshInfo, const MeshInfo3D& meshInfoProlongated, const InterpolationType interpolation, const BoundaryType boundary);

//=======================================
// C++ interface
//=======================================

class Prolongation1D {
public:

  const MeshInfo1D & _meshInfo;
  const MeshInfo1D & _meshInfoProlongated;
  const InterpolationType _interpolation;
  const BoundaryType _boundary;

  Prolongation1D( const MeshInfo1D & meshInfo, const MeshInfo1D& meshInfoProlongated,   
                  const InterpolationType interpolation, const BoundaryType boundary ) : 
                  _meshInfo(meshInfo), _meshInfoProlongated(meshInfoProlongated),
                  _interpolation ( interpolation ), _boundary( boundary ) {}

  torch::Tensor forward(const torch::Tensor &u ) const {
    CHECK_INPUT(u);
    return cuda_prolongate1d(u,_meshInfo,_meshInfoProlongated,_interpolation,_boundary);
  }

};


class Prolongation2D {
public:

  const MeshInfo2D & _meshInfo;
  const MeshInfo2D & _meshInfoProlongated;
  const InterpolationType _interpolation;
  const BoundaryType _boundary;

  Prolongation2D( const MeshInfo2D & meshInfo, const MeshInfo2D& meshInfoProlongated,
                  const InterpolationType interpolation, const BoundaryType boundary ) : 
                  _meshInfo(meshInfo), _meshInfoProlongated(meshInfoProlongated),
                  _interpolation ( interpolation ), _boundary( boundary ) {}

  torch::Tensor forward(const torch::Tensor &u) const {
    CHECK_INPUT(u);
    return cuda_prolongate2d(u,_meshInfo,_meshInfoProlongated,_interpolation,_boundary);
  }

  torch::Tensor forwardVectorField(const torch::Tensor &u) const {
    CHECK_INPUT(u);
    return cuda_prolongateVectorField2d(u,_meshInfo,_meshInfoProlongated,_interpolation,_boundary);
  }

  torch::Tensor forwardMatrixField(const torch::Tensor &u) const {
    CHECK_INPUT(u);
    return cuda_prolongateMatrixField2d(u,_meshInfo,_meshInfoProlongated,_interpolation,_boundary);
  }

};


class Prolongation3D {
public:

  const MeshInfo3D & _meshInfo;
  const MeshInfo3D & _meshInfoProlongated;
  const InterpolationType _interpolation;
  const BoundaryType _boundary;

  Prolongation3D( const MeshInfo3D & meshInfo, const MeshInfo3D& meshInfoProlongated,
                  const InterpolationType interpolation, const BoundaryType boundary ) : 
                  _meshInfo(meshInfo), _meshInfoProlongated(meshInfoProlongated),
                  _interpolation ( interpolation ), _boundary( boundary ) {}

  torch::Tensor forward(const torch::Tensor &u) const {
    CHECK_INPUT(u);
    return cuda_prolongate3d(u,_meshInfo,_meshInfoProlongated,_interpolation,_boundary);
  }

  torch::Tensor forwardVectorField(const torch::Tensor &u) const {
    CHECK_INPUT(u);
    return cuda_prolongateVectorField3d(u,_meshInfo,_meshInfoProlongated,_interpolation,_boundary);
  }

  torch::Tensor forwardMatrixField(const torch::Tensor &u) const {
    CHECK_INPUT(u);
    return cuda_prolongateMatrixField3d(u,_meshInfo,_meshInfoProlongated,_interpolation,_boundary);
  }

};

#endif