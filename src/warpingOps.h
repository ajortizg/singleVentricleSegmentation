#ifndef __WARPING_H_
#define __WARPING_H_

#include <torch/extension.h>
#include <vector>

//=======================================
// CUDA forward declarations
//=======================================

torch::Tensor cuda_warp1d_cubicSpline( const torch::Tensor &u, const torch::Tensor &phi);
torch::Tensor cuda_warp2d_cubicSpline( const torch::Tensor &u, const torch::Tensor &phi);
torch::Tensor cuda_warp3d_cubicSpline( const torch::Tensor &u, const torch::Tensor &phi);

//=======================================
// C++ interface
//=======================================
#define CHECK_CUDA(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



torch::Tensor warp1d_cubicSpline(
    const torch::Tensor &u,
    const torch::Tensor &phi)
{
  CHECK_INPUT(u);
  CHECK_INPUT(phi);

  return cuda_warp1d_cubicSpline(u,phi);
}

torch::Tensor warp2d_cubicSpline(
    const torch::Tensor &u,
    const torch::Tensor &phi)
{
  CHECK_INPUT(u);
  CHECK_INPUT(phi);

  return cuda_warp2d_cubicSpline(u,phi);
}

torch::Tensor warp3d_cubicSpline(
    const torch::Tensor &u,
    const torch::Tensor &phi)
{
  CHECK_INPUT(u);
  CHECK_INPUT(phi);

  return cuda_warp3d_cubicSpline(u,phi);
}

#endif