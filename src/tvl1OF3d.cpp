#include <torch/extension.h>
#include <vector>

//=======================================
// CUDA forward declarations
//=======================================
// void cuda_primal_update_step(
//     const torch::Tensor &u,
//     const torch::Tensor &p,
//     const torch::Tensor &ATq,
//     const float tau , //const torch::Tensor &tau, //
//     const float hz);

// // CUDA forward declarations
// void cuda_dual_update_step(
//     const torch::Tensor &p,
//     const torch::Tensor &u,
//     const torch::Tensor &sigma,
//     const float hz,
//     const float lamda);

// void cuda_prox_l2(
//     const torch::Tensor &q,
//     const torch::Tensor &sigma);

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

// torch::Tensor alternative_cuda_nabla3d_cd_forward(
//     const torch::Tensor &b);

//=======================================
// C++ interface
//=======================================
#define CHECK_CUDA(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// void primal_update_step(
//     const torch::Tensor &u,
//     const torch::Tensor &p,
//     const torch::Tensor &ATq,
//     const float tau, //const torch::Tensor &tau, /
//     const float hz)
// {
//   CHECK_INPUT(p);
//   CHECK_INPUT(u);
//   CHECK_INPUT(ATq);
//   //CHECK_INPUT(tau);

//   return cuda_primal_update_step(u, p, ATq, tau, hz);
// }

// void dual_update_step(
//     const torch::Tensor &p,
//     const torch::Tensor &u,
//     const torch::Tensor &sigma,
//     const float hz,
//     const float lamda)
// {
//   CHECK_INPUT(p);
//   CHECK_INPUT(u);

//   return cuda_dual_update_step(p, u, sigma, hz, lamda);
// }

// void prox_l2(
//     const torch::Tensor &q,
//     const torch::Tensor &sigma)
// {
//   CHECK_INPUT(q);

//   return cuda_prox_l2(q, sigma);
// }



torch::Tensor nabla1d_fd_forward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_nabla1d_fd_forward(b);
}

torch::Tensor divergence1d_fd_backward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_divergence1d_fd_backward(b);
}

torch::Tensor nabla2d_fd_forward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_nabla2d_fd_forward(b);
}

torch::Tensor divergence2d_fd_backward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_divergence2d_fd_backward(b);
}

torch::Tensor nabla3d_fd_forward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_nabla3d_fd_forward(b);
}

torch::Tensor divergence3d_fd_backward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_divergence3d_fd_backward(b);
}

torch::Tensor nabla1d_cd_forward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_nabla1d_cd_forward(b);
}

torch::Tensor divergence1d_cd_backward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_divergence1d_cd_backward(b);
}

torch::Tensor nabla2d_cd_forward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_nabla2d_cd_forward(b);
}

torch::Tensor divergence2d_cd_backward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_divergence2d_cd_backward(b);
}

torch::Tensor nabla3d_cd_forward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_nabla3d_cd_forward(b);
}

torch::Tensor divergence3d_cd_backward(
    const torch::Tensor &b)
{
  CHECK_INPUT(b);

  return cuda_divergence3d_cd_backward(b);
}

// torch::Tensor alternative_nabla3d_cd_forward(
//     const torch::Tensor &b)
// {
//   CHECK_INPUT(b);

//   return alternative_cuda_nabla3d_cd_forward(b);
// }

//=======================================
// python interface
//=======================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  // m.def("primal_step", &primal_update_step, "Update step for primal variable u");
  // m.def("dual_step", &dual_update_step, "Update step for dual variable p");
  // m.def("prox_l2", &prox_l2, "Proximal operator for L2 function");
  m.def("nabla1d_fd_forward", &nabla1d_fd_forward, "nabla in 1D with forward difference quotients");
  m.def("divergence1d_fd_backward", &divergence1d_fd_backward, "divergence in 1D with forward difference quotients");
  m.def("nabla2d_fd_forward", &nabla2d_fd_forward, "nabla in 2D with forward difference quotients");
  m.def("divergence2d_fd_backward", &divergence2d_fd_backward, "divergence in 2D with forward difference quotients");
  m.def("nabla3d_fd_forward", &nabla3d_fd_forward, "nabla in 3D with forward difference quotients");
  m.def("divergence3d_fd_backward", &divergence3d_fd_backward, "divergence in 3D with forward difference quotients");
  m.def("nabla1d_cd_forward", &nabla1d_cd_forward, "nabla in 1D with central difference quotients");
  m.def("divergence1d_cd_backward", &divergence1d_cd_backward, "divergence in 1D with central difference quotients");
  m.def("nabla2d_cd_forward", &nabla2d_cd_forward, "nabla in 2D with central difference quotients");
  m.def("divergence2d_cd_backward", &divergence2d_cd_backward, "divergence in 2D with central difference quotients");
  m.def("nabla3d_cd_forward", &nabla3d_cd_forward, "nabla in 3D with central difference quotients");
  m.def("divergence3d_cd_backward", &divergence3d_cd_backward, "divergence in 3D with central difference quotients");
  //m.def("alternative_nabla3d_cd_forward", &alternative_nabla3d_cd_forward, "3D central difference quotients for (X,Y,Z)-data");
}
