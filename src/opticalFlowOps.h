#ifndef __OPTICALFLOW_H_
#define __OPTICALFLOW_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"

//=======================================
// CUDA forward declarations
//=======================================

torch::Tensor cuda_TVL1OF_threshold( const torch::Tensor &u, const torch::Tensor &rho, 
                                     const torch::Tensor &I1_warped_gradx, const torch::Tensor &I1_warped_grady, const torch::Tensor &I1_warped_gradz,  
                                     const float LT,
                                     const MeshInfo3D &meshInfo);

//=======================================
// C++ interface
//=======================================

  torch::Tensor TVL1OF_threshold(const torch::Tensor &u, const torch::Tensor &rho, 
                                 const torch::Tensor &I1_warped_gradx, const torch::Tensor &I1_warped_grady, const torch::Tensor &I1_warped_gradz,
                                 const float LT,
                                 const MeshInfo3D &meshInfo){
    CHECK_INPUT(u);
    CHECK_INPUT(rho);
    CHECK_INPUT(I1_warped_gradx);
    CHECK_INPUT(I1_warped_grady);
    CHECK_INPUT(I1_warped_gradz);
    return cuda_TVL1OF_threshold(u,rho,I1_warped_gradx,I1_warped_grady,I1_warped_gradz,LT,meshInfo);
  }


// class Nabla1D_FD {
// public:

//   const MeshInfo1D & _meshInfo;

//   Nabla1D_FD( const MeshInfo1D & meshInfo ) : _meshInfo(meshInfo) {}

//   torch::Tensor forward(const torch::Tensor &b) const {
//     CHECK_INPUT(b);
//     return cuda_nabla1d_fd_forward(b,_meshInfo);
//   }
//   torch::Tensor backward(const torch::Tensor &b) const{
//     CHECK_INPUT(b);
//     return cuda_divergence1d_fd_backward(b,_meshInfo);
//   }
// };

// class Nabla2D_FD {
// public:
 
//   const MeshInfo2D & _meshInfo;

//   Nabla2D_FD( const MeshInfo2D & meshInfo ) : _meshInfo(meshInfo) {}

//   torch::Tensor forward(const torch::Tensor &b) const {
//     CHECK_INPUT(b);
//     return cuda_nabla2d_fd_forward(b,_meshInfo);
//   }
//   torch::Tensor backward(const torch::Tensor &b) const{
//     CHECK_INPUT(b);
//     return cuda_divergence2d_fd_backward(b,_meshInfo);
//   }
// };

// class Nabla3D_FD {
// public:

//   const MeshInfo3D & _meshInfo;

//   Nabla3D_FD( const MeshInfo3D & meshInfo ) : _meshInfo(meshInfo) {}

//   torch::Tensor forward(const torch::Tensor &b) const {
//     CHECK_INPUT(b);
//     return cuda_nabla3d_fd_forward(b,_meshInfo);
//   }
//   torch::Tensor backward(const torch::Tensor &b) const{
//     CHECK_INPUT(b);
//     return cuda_divergence3d_fd_backward(b,_meshInfo);
//   }
// };



#endif