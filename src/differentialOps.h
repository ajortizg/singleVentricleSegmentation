#ifndef __DIFFERENTIALOPS_H_
#define __DIFFERENTIALOPS_H_

#include <torch/extension.h>
#include <vector>

//=======================================
// CUDA forward declarations
//=======================================

torch::Tensor cuda_nabla1d_fd_forward( const torch::Tensor &b);
torch::Tensor cuda_divergence1d_fd_backward( const torch::Tensor &b);

torch::Tensor cuda_nabla2d_fd_forward( const torch::Tensor &b);
torch::Tensor cuda_divergence2d_fd_backward( const torch::Tensor &b);

torch::Tensor cuda_nabla3d_fd_forward( const torch::Tensor &b);
torch::Tensor cuda_divergence3d_fd_backward( const torch::Tensor &b);

torch::Tensor cuda_nabla1d_cd_forward( const torch::Tensor &b);
torch::Tensor cuda_divergence1d_cd_backward( const torch::Tensor &b);

torch::Tensor cuda_nabla2d_cd_forward( const torch::Tensor &b);
torch::Tensor cuda_divergence2d_cd_backward( const torch::Tensor &b);

torch::Tensor cuda_nabla3d_cd_forward( const torch::Tensor &b);
torch::Tensor cuda_divergence3d_cd_backward( const torch::Tensor &b);

//=======================================
// C++ interface
//=======================================
#define CHECK_CUDA(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


class Nabla1D_FD {
public:
  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla1d_fd_forward(b);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence1d_fd_backward(b);
  }
};

class Nabla2D_FD {
public:
  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla2d_fd_forward(b);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence2d_fd_backward(b);
  }
};

class Nabla3D_FD {
public:
  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla3d_fd_forward(b);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence3d_fd_backward(b);
  }
};


class Nabla1D_CD {
public:
  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla1d_cd_forward(b);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence1d_cd_backward(b);
  }
};

class Nabla2D_CD {
public:
  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla2d_cd_forward(b);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence2d_cd_backward(b);
  }
};

class Nabla3D_CD {
public:
  torch::Tensor forward(const torch::Tensor &b) const {
    CHECK_INPUT(b);
    return cuda_nabla3d_cd_forward(b);
  }
  torch::Tensor backward(const torch::Tensor &b) const{
    CHECK_INPUT(b);
    return cuda_divergence3d_cd_backward(b);
  }
};


#endif