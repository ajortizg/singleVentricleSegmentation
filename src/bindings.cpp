#include <pybind11/pybind11.h>
#include "differentialOps.h"
#include "prolongationOps.h"
#include "warpingOps.h"

namespace py = pybind11;


//=======================================
// python interface
//=======================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

  py::class_<MeshInfo1D>(m,"MeshInfo1D")
        .def(py::init<const int, const float>());

  py::class_<MeshInfo2D>(m,"MeshInfo2D")
        .def(py::init<const int, const int, const float, const float>());

  py::class_<MeshInfo3D>(m,"MeshInfo3D")
        .def(py::init<const int, const int, const int, const float, const float, const float>());

  //=======================================
  // differentialOps
  //======================================= 
  py::class_<Nabla1D_FD>(m,"Nabla1D_FD")
        .def(py::init<const MeshInfo1D&>())
        .def("forward", &Nabla1D_FD::forward)
        .def("backward", &Nabla1D_FD::backward);

  py::class_<Nabla2D_FD>(m,"Nabla2D_FD")
        .def(py::init<const MeshInfo2D&>())
        .def("forward", &Nabla2D_FD::forward)
        .def("backward", &Nabla2D_FD::backward);

  py::class_<Nabla3D_FD>(m,"Nabla3D_FD")
        .def(py::init<const MeshInfo3D&>())
        .def("forward", &Nabla3D_FD::forward)
        .def("backward", &Nabla3D_FD::backward);

  py::class_<Nabla1D_CD>(m,"Nabla1D_CD")
        .def(py::init<const MeshInfo1D&>())
        .def("forward", &Nabla1D_CD::forward)
        .def("backward", &Nabla1D_CD::backward);

  py::class_<Nabla2D_CD>(m,"Nabla2D_CD")
        .def(py::init<const MeshInfo2D&>())
        .def("forward", &Nabla2D_CD::forward)
        .def("backward", &Nabla2D_CD::backward);

  py::class_<Nabla3D_CD>(m,"Nabla3D_CD")
        .def(py::init<const MeshInfo3D&>())
        .def("forward", &Nabla3D_CD::forward)
        .def("backward", &Nabla3D_CD::backward);


  //=======================================
  // interpolation
  //======================================= 
  py::enum_<InterpolationType>(m, "InterpolationType")
    .value("INTERPOLATE_LINEAR", InterpolationType::INTERPOLATE_LINEAR)
    .value("INTERPOLATE_CUBIC_HERMITESPLINE", InterpolationType::INTERPOLATE_CUBIC_HERMITESPLINE)
    .value("INTERPOLATE_CUBIC_BSPLINE", InterpolationType::INTERPOLATE_CUBIC_BSPLINE)
    .export_values();

  //=======================================
  // warping
  //======================================= 
  py::class_<Warping1D>(m,"Warping1D")
      .def(py::init<const MeshInfo1D&>())
      .def("forward", &Warping1D::forward);

  py::class_<Warping2D>(m,"Warping2D")
      .def(py::init<const MeshInfo2D&>())
      .def("forward", &Warping2D::forward);

  py::class_<Warping3D>(m,"Warping3D")
      .def(py::init<const MeshInfo3D&>())
      .def("forward", &Warping3D::forward);

  //=======================================
  // prolongation
  //======================================= 
  py::class_<Prolongation1D>(m,"Prolongation1D")
      .def(py::init<const MeshInfo1D&,const MeshInfo1D&>())
      .def("forward", &Prolongation1D::forward);

  py::class_<Prolongation2D>(m,"Prolongation2D")
      .def(py::init<const MeshInfo2D&,const MeshInfo2D&>())
      .def("forward", &Prolongation2D::forward);

  py::class_<Prolongation3D>(m,"Prolongation3D")
      .def(py::init<const MeshInfo3D&,const MeshInfo3D&>())
      .def("forward", &Prolongation3D::forward);


}
