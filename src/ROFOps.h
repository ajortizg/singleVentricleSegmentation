#ifndef __ROFOPS_H_
#define __ROFOPS_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"

//=======================================
// CUDA forward declarations
//=======================================

torch::Tensor cuda_ROF2D_proxPrimal( const torch::Tensor &primalVariable,
                                     const torch::Tensor &inputImage,
                                     const float primalStepSize_tau, 
                                     const float primalFctWeight_Matching,
                                     const MeshInfo2D &meshInfo);
                                      
torch::Tensor cuda_ROF2D_proxDual( const torch::Tensor &dualVariable, 
                                   const float dualStepSize_sigma,
                                   const float dualFctWeight_TV,
                                   const MeshInfo2D &meshInfo);

torch::Tensor cuda_ROF3D_proxPrimal( const torch::Tensor &primalVariable,
                                     const torch::Tensor &inputImage,
                                     const float primalStepSize_tau, 
                                     const float primalFctWeight_Matching,
                                     const MeshInfo3D &meshInfo);
                                      
torch::Tensor cuda_ROF3D_proxDual( const torch::Tensor &dualVariable, 
                                   const float dualStepSize_sigma,
                                   const float dualFctWeight_TV,
                                   const MeshInfo3D &meshInfo);



//=======================================
// C++ interface
//=======================================

torch::Tensor ROF2D_proxPrimal( const torch::Tensor &primalVariable,
                                const torch::Tensor &inputImage,
                                const float primalStepSize_tau, 
                                const float primalFctWeight_Matching,
                                const MeshInfo2D &meshInfo){

  CHECK_INPUT(primalVariable);     
  return cuda_ROF2D_proxPrimal( primalVariable, inputImage, primalStepSize_tau, primalFctWeight_Matching, meshInfo);                          
};


torch::Tensor ROF2D_proxDual( const torch::Tensor &dualVariable, 
                              const float dualStepSize_sigma,
                              const float dualFctWeight_TV,
                              const MeshInfo2D &meshInfo){

 CHECK_INPUT(dualVariable);    
 return cuda_ROF2D_proxDual(dualVariable,dualStepSize_sigma,dualFctWeight_TV,meshInfo);                                
};


torch::Tensor ROF3D_proxPrimal( const torch::Tensor &primalVariable,
                                const torch::Tensor &inputImage,
                                const float primalStepSize_tau, 
                                const float primalFctWeight_Matching,
                                const MeshInfo3D &meshInfo){

  CHECK_INPUT(primalVariable);     
  return cuda_ROF3D_proxPrimal( primalVariable, inputImage, primalStepSize_tau, primalFctWeight_Matching, meshInfo);                          
};


torch::Tensor ROF3D_proxDual( const torch::Tensor &dualVariable, 
                              const float dualStepSize_sigma,
                              const float dualFctWeight_TV,
                              const MeshInfo3D &meshInfo){

 CHECK_INPUT(dualVariable);    
 return cuda_ROF3D_proxDual(dualVariable,dualStepSize_sigma,dualFctWeight_TV,meshInfo);                                
};


#endif