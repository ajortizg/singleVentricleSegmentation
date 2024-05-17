#ifndef __DIFFERENTIALOPSFD_H_
#define __DIFFERENTIALOPSFD_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"

//=======================================
// CUDA forward declarations
//=======================================

torch::Tensor cuda_nabla1d_fd_forward( const torch::Tensor &b, const MeshInfo1D &meshInfo);
torch::Tensor cuda_divergence1d_fd_backward( const torch::Tensor &b, const MeshInfo1D &meshInfo);

torch::Tensor cuda_nabla2d_fd_forward( const torch::Tensor &b, const MeshInfo2D &meshInfo);
torch::Tensor cuda_divergence2d_fd_backward( const torch::Tensor &b, const MeshInfo2D &meshInfo);

torch::Tensor cuda_nabla3d_fd_forward( const torch::Tensor &b, const MeshInfo3D &meshInfo);
torch::Tensor cuda_divergence3d_fd_backward( const torch::Tensor &b, const MeshInfo3D &meshInfo);

//=======================================
// C++ interface
//=======================================

class Nabla1D_FD {
public:

  const MeshInfo1D & _meshInfo;

  Nabla1D_FD( const MeshInfo1D & meshInfo ) : _meshInfo(meshInfo) {}

  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla1d_fd_forward(b,_meshInfo);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence1d_fd_backward(b,_meshInfo);
  }
};

class Nabla2D_FD {
public:
 
  const MeshInfo2D & _meshInfo;

  Nabla2D_FD( const MeshInfo2D & meshInfo ) : _meshInfo(meshInfo) {}

  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla2d_fd_forward(b,_meshInfo);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence2d_fd_backward(b,_meshInfo);
  }
};

class Nabla3D_FD {
public:

  const MeshInfo3D & _meshInfo;

  Nabla3D_FD( const MeshInfo3D & meshInfo ) : _meshInfo(meshInfo) {}

  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla3d_fd_forward(b,_meshInfo);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence3d_fd_backward(b,_meshInfo);
  }
};


#endif