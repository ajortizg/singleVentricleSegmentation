#include <pybind11/pybind11.h>
#include "differentialOps.h"
#include "anisotropicDifferentialOps.h"
#include "prolongationOps.h"
#include "rotationOps.h"
#include "warpingOps.h"
#include "warpingOpsCNN.h"
#include "opticalFlowOps.h"
#include "ROFOps.h"

namespace py = pybind11;

//=======================================
// python interface
//=======================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  //=======================================
  // mesh
  //======================================= 
  py::class_<MeshInfo1D>(m,"MeshInfo1D")
        .def(py::init<const int, const float>())
        .def("getNX", &MeshInfo1D::getNX)
        .def("getLX", &MeshInfo1D::getLX)
        .def("gethX", &MeshInfo1D::gethX);

  py::class_<MeshInfo2D>(m,"MeshInfo2D")
        .def(py::init<const int, const int, const float, const float>())
        .def("getNX", &MeshInfo2D::getNX)
        .def("getLX", &MeshInfo2D::getLX)
        .def("gethX", &MeshInfo2D::gethX)
        .def("getNY", &MeshInfo2D::getNY)
        .def("getLY", &MeshInfo2D::getLY)
        .def("gethY", &MeshInfo2D::gethY);

  py::class_<MeshInfo3D>(m,"MeshInfo3D")
        .def(py::init<const int, const int, const int, const float, const float, const float>())
        .def("getNX", &MeshInfo3D::getNX)
        .def("getLX", &MeshInfo3D::getLX)
        .def("gethX", &MeshInfo3D::gethX)
        .def("getNY", &MeshInfo3D::getNY)
        .def("getLY", &MeshInfo3D::getLY)
        .def("gethY", &MeshInfo3D::gethY)
        .def("getNZ", &MeshInfo3D::getNZ)
        .def("getLZ", &MeshInfo3D::getLZ)
        .def("gethZ", &MeshInfo3D::gethZ);

  //=======================================
  // boundary
  //======================================= 
  py::enum_<BoundaryType>(m, "BoundaryType")
    .value("BOUNDARY_NEAREST", BoundaryType::BOUNDARY_NEAREST)
    .value("BOUNDARY_MIRROR", BoundaryType::BOUNDARY_MIRROR)
    .value("BOUNDARY_REFLECT", BoundaryType::BOUNDARY_REFLECT)
    .export_values();

  //=======================================
  // differentialOps
  //======================================= 
  py::class_<Nabla1D_CD>(m,"Nabla1D_CD")
        .def(py::init<const MeshInfo1D&, const BoundaryType>())
        .def("forward", &Nabla1D_CD::forward)
        .def("backward", &Nabla1D_CD::backward);

  py::class_<Nabla2D_CD>(m,"Nabla2D_CD")
        .def(py::init<const MeshInfo2D&, const BoundaryType>())
        .def("forward", &Nabla2D_CD::forward)
        .def("backward", &Nabla2D_CD::backward)
        .def("forwardVectorField", &Nabla2D_CD::forwardVectorField)
        .def("backwardVectorField", &Nabla2D_CD::backwardVectorField);

  py::class_<Nabla3D_CD>(m,"Nabla3D_CD")
        .def(py::init<const MeshInfo3D&, const BoundaryType>())
        .def("forward", &Nabla3D_CD::forward)
        .def("backward", &Nabla3D_CD::backward)
        .def("forwardVectorField", &Nabla3D_CD::forwardVectorField)
        .def("backwardVectorField", &Nabla3D_CD::backwardVectorField);

  //=======================================
  // anisotropicDifferentialOps
  //======================================= 

  py::class_<AnisotropicNabla2D>(m,"AnisotropicNabla2D")
        .def(py::init<const MeshInfo2D&, const float, const float>())
        .def("computeTangentVecs", &AnisotropicNabla2D::computeTangentVecs)
        .def("forwardVectorField", &AnisotropicNabla2D::forwardVectorField)
        .def("backwardVectorField", &AnisotropicNabla2D::backwardVectorField);

  py::class_<AnisotropicNabla3D>(m,"AnisotropicNabla3D")
        .def(py::init<const MeshInfo3D&, const float, const float>())
        .def("computeTangentVecs", &AnisotropicNabla3D::computeTangentVecs)
        .def("forwardVectorField", &AnisotropicNabla3D::forwardVectorField)
        .def("backwardVectorField", &AnisotropicNabla3D::backwardVectorField);


  //=======================================
  // interpolation
  //======================================= 
  py::enum_<InterpolationType>(m, "InterpolationType")
    .value("INTERPOLATE_NEAREST", InterpolationType::INTERPOLATE_NEAREST)
    .value("INTERPOLATE_LINEAR", InterpolationType::INTERPOLATE_LINEAR)
    .value("INTERPOLATE_CUBIC_HERMITESPLINE", InterpolationType::INTERPOLATE_CUBIC_HERMITESPLINE)
    //.value("INTERPOLATE_CUBIC_BSPLINE", InterpolationType::INTERPOLATE_CUBIC_BSPLINE)
    .export_values();

  //=======================================
  // rotation
  //======================================= 
  py::class_<Rotation3D>(m,"Rotation3D")
      .def(py::init<const MeshInfo3D&, const InterpolationType, const BoundaryType>())
      .def("forward", &Rotation3D::forward)
      .def("forwardVectorField", &Rotation3D::forwardVectorField)
      .def("forwardMatrixField", &Rotation3D::forwardMatrixField);

  //=======================================
  // warping
  //======================================= 
  py::class_<Warping1D>(m,"Warping1D")
      .def(py::init<const MeshInfo1D&, const InterpolationType, const BoundaryType>())
      .def("forward", &Warping1D::forward)
      .def("backward", &Warping1D::backward);

  py::class_<Warping2D>(m,"Warping2D")
      .def(py::init<const MeshInfo2D&, const InterpolationType, const BoundaryType>())
      .def("forward", &Warping2D::forward)
      .def("backward", &Warping2D::backward)
      .def("forwardVectorField", &Warping2D::forwardVectorField);

  py::class_<Warping3D>(m,"Warping3D")
      .def(py::init<const MeshInfo3D&, const InterpolationType, const BoundaryType>())
      .def("forward", &Warping3D::forward)
      .def("backward", &Warping3D::backward)
      .def("forwardVectorField", &Warping3D::forwardVectorField);

  //=======================================
  // warping cnn
  //======================================= 
  py::class_<WarpingCNN3D>(m,"WarpingCNN3D")
      .def(py::init<const MeshInfo3D&, const InterpolationType, const BoundaryType>())
      .def("forward", &WarpingCNN3D::forward);

  //=======================================
  // prolongation
  //======================================= 
  py::class_<Prolongation1D>(m,"Prolongation1D")
      .def(py::init<const MeshInfo1D&,const MeshInfo1D&, const InterpolationType, const BoundaryType>())
      .def("forward", &Prolongation1D::forward);

  py::class_<Prolongation2D>(m,"Prolongation2D")
      .def(py::init<const MeshInfo2D&,const MeshInfo2D&, const InterpolationType, const BoundaryType>())
      .def("forward", &Prolongation2D::forward)
      .def("forwardVectorField", &Prolongation2D::forwardVectorField)
      .def("forwardMatrixField", &Prolongation2D::forwardMatrixField);

  py::class_<Prolongation3D>(m,"Prolongation3D")
      .def(py::init<const MeshInfo3D&,const MeshInfo3D&, const InterpolationType, const BoundaryType>())
      .def("forward", &Prolongation3D::forward)
      .def("forwardVectorField", &Prolongation3D::forwardVectorField)
      .def("forwardMatrixField", &Prolongation3D::forwardMatrixField);

  //=======================================
  // optical flow
  //=======================================
  m.def("TVL1OF2D_proxPrimal", &TVL1OF2D_proxPrimal, "primal prox step for OF in 2D");
  m.def("TVL1OF2D_proxDual", &TVL1OF2D_proxDual, "dual prox step for OF in 2D");
  m.def("TVL1OF3D_proxPrimal", &TVL1OF3D_proxPrimal, "primal prox step for OF in 3D");
  m.def("TVL1OF3D_proxDual", &TVL1OF3D_proxDual, "dual prox step for OF in 3D");
  m.def("TVL1SymOF3D_proxPrimal", &TVL1SymOF3D_proxPrimal, "primal prox step for symmetrized OF in 3D");


  //=======================================
  // ROF
  //=======================================
  // py::class_<ROF2D>(m,"ROF2D")
  //       .def(py::init<const MeshInfo2D&, const torch::Tensor&, const float, const float>())
  //       .def("proxPrimal", &ROF2D::proxPrimal)
  //       .def("proxDual", &ROF2D::proxDual);
  py::class_<ROF2D>(m,"ROF2D")
        .def(py::init<const MeshInfo2D&, const float, const float>())
        .def("proxPrimal", &ROF2D::proxPrimal)
        .def("proxDual", &ROF2D::proxDual);

  m.def("ROF3D_proxPrimal", &ROF3D_proxPrimal, "ROF primal prox step in 3D");
  m.def("ROF3D_proxDual", &ROF3D_proxDual, "ROF dual prox step in 3D");
  //TODO 
      //   py::class_<ROF3D>(m,"ROF3D")
      //         .def(py::init<const MeshInfo2D&, const float, const float>())
      //         .def("proxPrimal", &ROF2D::proxPrimal)
      //         .def("proxDual", &ROF2D::proxDual);


}
