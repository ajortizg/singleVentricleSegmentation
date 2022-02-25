#include <pybind11/pybind11.h>
#include "differentialOps.h"
#include "warping.h"

namespace py = pybind11;


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


  //=======================================
  // from warping
  //=======================================
  m.def("warp1d_cubicSpline", &warp1d_cubicSpline, "warp in 1d with cubic spline");
  m.def("warp2d_cubicSpline", &warp2d_cubicSpline, "warp in 2d with cubic spline");
  m.def("warp3d_cubicSpline", &warp3d_cubicSpline, "warp in 3d with cubic spline");

}
