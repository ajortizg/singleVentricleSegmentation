#ifndef __OPTICALFLOW_H_
#define __OPTICALFLOW_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"

//=======================================
// CUDA forward declarations
//=======================================


//old
// torch::Tensor cuda_TVL1OF_threshold( const torch::Tensor &u, const torch::Tensor &rho, 
//                                      const torch::Tensor &I1_warped_gradx, const torch::Tensor &I1_warped_grady, const torch::Tensor &I1_warped_gradz,  
//                                      const float LT,
//                                      const MeshInfo3D &meshInfo);


// void cuda_TVL1OF_updateDualVariable(torch::Tensor &u, const torch::Tensor &v, torch::Tensor &p,
//                                  const float TAU, const float THETA
//                                  const MeshInfo3D &meshInfo);

//new: CP

torch::Tensor cuda_TVL1OF2D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad, 
                                      const float weightNorm,
                                      const MeshInfo2D &meshInfo);
                                      
torch::Tensor cuda_TVL1OF2D_proxDual( const torch::Tensor &dualVariable, 
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo2D &meshInfo);


torch::Tensor cuda_TVL1OF3D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad,  
                                      const float weightNorm,
                                      const MeshInfo3D &meshInfo);
                                      
torch::Tensor cuda_TVL1OF3D_proxDual( const torch::Tensor &dualVariable, 
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo3D &meshInfo);



//=======================================
// C++ interface
//=======================================

//old
  // torch::Tensor TVL1OF_threshold(const torch::Tensor &u, const torch::Tensor &rho, 
  //                                const torch::Tensor &I1_warped_gradx, const torch::Tensor &I1_warped_grady, const torch::Tensor &I1_warped_gradz,
  //                                const float LT,
  //                                const MeshInfo3D &meshInfo){
  //   CHECK_INPUT(u);
  //   CHECK_INPUT(rho);
  //   CHECK_INPUT(I1_warped_gradx);
  //   CHECK_INPUT(I1_warped_grady);
  //   CHECK_INPUT(I1_warped_gradz);
  //   return cuda_TVL1OF_threshold(u,rho,I1_warped_gradx,I1_warped_grady,I1_warped_gradz,LT,meshInfo);
  // }

  // void TVL1OF_updateDualVariable(torch::Tensor &u, const torch::Tensor &v, torch::Tensor &p,
  //                                const float TAU, const float THETA
  //                                const MeshInfo3D &meshInfo){
  //   CHECK_INPUT(u);
  //   CHECK_INPUT(v);
  //   CHECK_INPUT(p);
  //   cuda_TVL1OF_updateDualVariable(u,v,p,TAU,THETA,meshInfo);
  // }


//new: CP

// torch::Tensor TVL1OF_proxPrimal( const torch::Tensor &primalVariable,
//                                       const float primalStepSize_tau, 
//                                       const float primalFctWeight_Matching,
//                                       const torch::Tensor &rho, 
//                                       const torch::Tensor &I1_warped_gradx, const torch::Tensor &I1_warped_grady, const torch::Tensor &I1_warped_gradz,  
//                                       const float weightNorm,
//                                       const MeshInfo3D &meshInfo){

//   CHECK_INPUT(primalVariable);     
//   return cuda_TVL1OF_proxPrimal( primalVariable, primalStepSize_tau, primalFctWeight_Matching, rho, I1_warped_gradx, I1_warped_grady, I1_warped_gradz, weightNorm, meshInfo);                          
// };



torch::Tensor TVL1OF2D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad,
                                      const float weightNorm,
                                      const MeshInfo2D &meshInfo){

  CHECK_INPUT(primalVariable);     
  return cuda_TVL1OF2D_proxPrimal( primalVariable, primalStepSize_tau, primalFctWeight_Matching, rho, I1_warped_grad, weightNorm, meshInfo);                          
};


torch::Tensor TVL1OF2D_proxDual( const torch::Tensor &dualVariable, 
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo2D &meshInfo){

 CHECK_INPUT(dualVariable);    
 return cuda_TVL1OF2D_proxDual(dualVariable,dualStepSize_sigma,dualFctWeight_TV,meshInfo);                                
};


torch::Tensor TVL1OF3D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad,
                                      const float weightNorm,
                                      const MeshInfo3D &meshInfo){

  CHECK_INPUT(primalVariable);     
  return cuda_TVL1OF3D_proxPrimal( primalVariable, primalStepSize_tau, primalFctWeight_Matching, rho, I1_warped_grad, weightNorm, meshInfo);                          
};


torch::Tensor TVL1OF3D_proxDual( const torch::Tensor &dualVariable, 
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo3D &meshInfo){

 CHECK_INPUT(dualVariable);    
 return cuda_TVL1OF3D_proxDual(dualVariable,dualStepSize_sigma,dualFctWeight_TV,meshInfo);                                
};


#endif