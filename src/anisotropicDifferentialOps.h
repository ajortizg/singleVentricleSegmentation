#ifndef __ANISOTROPICDIFFERENTIALOPS_H_
#define __ANISOTROPICDIFFERENTIALOPS_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"

//=======================================
// CUDA forward declarations
//=======================================

std::vector<torch::Tensor> cuda_AnisotropicNabla2d_computeTangentVecs( const torch::Tensor &NablaInputImage, const MeshInfo2D &meshInfo, const float alpha, const float beta);
torch::Tensor cuda_AnisotropicNabla2d_forwardVectorField( const torch::Tensor &Input, const MeshInfo2D &meshInfo, const torch::Tensor &scalars,  const torch::Tensor &normals,  const torch::Tensor &tangents);

std::vector<torch::Tensor> cuda_AnisotropicNabla3d_computeTangentVecs( const torch::Tensor &NablaInputImage, const MeshInfo3D &meshInfo, const float alpha, const float beta);
torch::Tensor cuda_AnisotropicNabla3d_forwardVectorField( const torch::Tensor &Input, const MeshInfo3D &meshInfo, const torch::Tensor &scalars,  const torch::Tensor &normals,  const torch::Tensor &tangents1, const torch::Tensor &tangents2);

//=======================================
// C++ interface
//=======================================

class AnisotropicNabla2D {
public:

  const MeshInfo2D & _meshInfo;
  const float _alpha;
  const float _beta;

  AnisotropicNabla2D( const MeshInfo2D & meshInfo, const float alpha, const float beta ) : 
   _meshInfo(meshInfo), _alpha( alpha), _beta(beta) {}

  std::vector<torch::Tensor> computeTangentVecs(const torch::Tensor &NablaInputImage) const {
    CHECK_INPUT(NablaInputImage);
    return cuda_AnisotropicNabla2d_computeTangentVecs(NablaInputImage,_meshInfo,_alpha,_beta);
  }

  torch::Tensor forwardVectorField(const torch::Tensor &b, const torch::Tensor &scalars, const torch::Tensor &normals, const torch::Tensor &tangents) const {
    CHECK_INPUT(b);
    CHECK_INPUT(scalars);
    CHECK_INPUT(normals);
    CHECK_INPUT(tangents);
    return cuda_AnisotropicNabla2d_forwardVectorField(b,_meshInfo,scalars,normals,tangents);
  }
  torch::Tensor backwardVectorField(const torch::Tensor &b, const torch::Tensor &scalars, const torch::Tensor &normals, const torch::Tensor &tangents) const{
    CHECK_INPUT(b);
    CHECK_INPUT(scalars);
    CHECK_INPUT(normals);
    CHECK_INPUT(tangents);
    return cuda_AnisotropicNabla2d_forwardVectorField(b,_meshInfo,scalars,normals,tangents);
  }
};


class AnisotropicNabla3D {
public:

  const MeshInfo3D & _meshInfo;
  const float _alpha;
  const float _beta;

  AnisotropicNabla3D( const MeshInfo3D & meshInfo, const float alpha, const float beta ) : 
   _meshInfo(meshInfo), _alpha( alpha), _beta(beta) {}

  std::vector<torch::Tensor> computeTangentVecs(const torch::Tensor &NablaInputImage) const {
    CHECK_INPUT(NablaInputImage);
    return cuda_AnisotropicNabla3d_computeTangentVecs(NablaInputImage,_meshInfo,_alpha,_beta);
  }

  torch::Tensor forwardVectorField(const torch::Tensor &b, const torch::Tensor &scalars, const torch::Tensor &normals, const torch::Tensor &tangents1, const torch::Tensor &tangents2) const {
    CHECK_INPUT(b);
    CHECK_INPUT(scalars);
    CHECK_INPUT(normals);
    CHECK_INPUT(tangents1);
    CHECK_INPUT(tangents2);
    return cuda_AnisotropicNabla3d_forwardVectorField(b,_meshInfo,scalars,normals,tangents1,tangents2);
  }
  torch::Tensor backwardVectorField(const torch::Tensor &b, const torch::Tensor &scalars, const torch::Tensor &normals, const torch::Tensor &tangents1, const torch::Tensor &tangents2) const{
    CHECK_INPUT(b);
    CHECK_INPUT(scalars);
    CHECK_INPUT(normals);
    CHECK_INPUT(tangents1);
    CHECK_INPUT(tangents2);
    return cuda_AnisotropicNabla3d_forwardVectorField(b,_meshInfo,scalars,normals,tangents1,tangents2);
  }
};

#endif