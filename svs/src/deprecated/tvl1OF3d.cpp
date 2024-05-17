#include <torch/extension.h>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;


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



// ======================================================
// old: functions: (remove, since they can be called by classes)
// ===================================================

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




// ======================================================
// new: classes
// ===================================================

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


//=======================================
// python interface
//=======================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

  py::class_<Nabla1D_FD>(m,"Nabla1D_FD")
        .def(py::init<>())
        .def("forward", &Nabla1D_FD::forward)
        .def("backward", &Nabla1D_FD::backward);

  py::class_<Nabla2D_FD>(m,"Nabla2D_FD")
        .def(py::init<>())
        .def("forward", &Nabla2D_FD::forward)
        .def("backward", &Nabla2D_FD::backward);

  py::class_<Nabla3D_FD>(m,"Nabla3D_FD")
        .def(py::init<>())
        .def("forward", &Nabla3D_FD::forward)
        .def("backward", &Nabla3D_FD::backward);

  py::class_<Nabla1D_CD>(m,"Nabla1D_CD")
        .def(py::init<>())
        .def("forward", &Nabla1D_CD::forward)
        .def("backward", &Nabla1D_CD::backward);

  py::class_<Nabla2D_CD>(m,"Nabla2D_CD")
        .def(py::init<>())
        .def("forward", &Nabla2D_CD::forward)
        .def("backward", &Nabla2D_CD::backward);

  py::class_<Nabla3D_CD>(m,"Nabla3D_CD")
        .def(py::init<>())
        .def("forward", &Nabla3D_CD::forward)
        .def("backward", &Nabla3D_CD::backward);

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
