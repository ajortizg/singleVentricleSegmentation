#ifndef __OPTICALFLOW_H_
#define __OPTICALFLOW_H_

#include <torch/extension.h>
#include <vector>
#include "coreDefines.h"

//=======================================
// CUDA forward declarations
//=======================================

// float cuda_TVL1OF2D_PrimalFct( const float primalFctWeight_Matching,
//                                const torch::Tensor &rho, 
//                                const MeshInfo2D &meshInfo);

torch::Tensor cuda_TVL1OF2D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad, 
                                      const MeshInfo2D &meshInfo);
                                      
torch::Tensor cuda_TVL1OF2D_proxDual( const torch::Tensor &dualVariable, 
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo2D &meshInfo);


// torch::Tensor cuda_TVL1OF3D_PrimalFct( const float primalFctWeight_Matching,
//                                const torch::Tensor &rho, 
//                                const MeshInfo3D &meshInfo);

torch::Tensor cuda_TVL1OF3D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad,  
                                      const MeshInfo3D &meshInfo);
                                      
torch::Tensor cuda_TVL1OF3D_proxDual( const torch::Tensor &dualVariable, 
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo3D &meshInfo);


torch::Tensor cuda_TVL1SymOF3D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho_const_l, const torch::Tensor &rho_vec_l,
                                      const torch::Tensor &rho_const_r, const torch::Tensor &rho_vec_r,
                                      const MeshInfo3D &meshInfo);



//=======================================
// C++ interface
//=======================================

// float TVL1OF2D_PrimalFct( const float primalFctWeight_Matching,
//                           const torch::Tensor &rho,
//                           const MeshInfo2D &meshInfo ){
//   CHECK_INPUT(rho);     
//   return cuda_TVL1OF2D_PrimalFct( primalFctWeight_Matching, rho, meshInfo);                          
// };

torch::Tensor TVL1OF2D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad,
                                      const MeshInfo2D &meshInfo){

  CHECK_INPUT(primalVariable);     
  return cuda_TVL1OF2D_proxPrimal( primalVariable, primalStepSize_tau, primalFctWeight_Matching, rho, I1_warped_grad, meshInfo);                          
};


torch::Tensor TVL1OF2D_proxDual( const torch::Tensor &dualVariable, 
                                    const float dualStepSize_sigma,
                                    const float dualFctWeight_TV,
                                    const MeshInfo2D &meshInfo){

 CHECK_INPUT(dualVariable);    
 return cuda_TVL1OF2D_proxDual(dualVariable,dualStepSize_sigma,dualFctWeight_TV,meshInfo);                                
};



// torch::Tensor TVL1OF3D_PrimalFct( const float primalFctWeight_Matching,
//                                   const torch::Tensor &rho,
//                                   const MeshInfo3D &meshInfo ){
//   CHECK_INPUT(rho);     
//   return cuda_TVL1OF3D_PrimalFct( primalFctWeight_Matching, rho, meshInfo);                          
// };

torch::Tensor TVL1OF3D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho, 
                                      const torch::Tensor &I1_warped_grad,
                                      const MeshInfo3D &meshInfo){

  CHECK_INPUT(primalVariable);     
  return cuda_TVL1OF3D_proxPrimal( primalVariable, primalStepSize_tau, primalFctWeight_Matching, rho, I1_warped_grad, meshInfo);                          
};


torch::Tensor TVL1OF3D_proxDual( const torch::Tensor &dualVariable, 
                                 const float dualStepSize_sigma,
                                 const float dualFctWeight_TV,
                                 const MeshInfo3D &meshInfo){

 CHECK_INPUT(dualVariable);    
 return cuda_TVL1OF3D_proxDual(dualVariable,dualStepSize_sigma,dualFctWeight_TV,meshInfo);                                
};


torch::Tensor TVL1SymOF3D_proxPrimal( const torch::Tensor &primalVariable,
                                      const float primalStepSize_tau, 
                                      const float primalFctWeight_Matching,
                                      const torch::Tensor &rho_const_l, const torch::Tensor &rho_vec_l,
                                      const torch::Tensor &rho_const_r, const torch::Tensor &rho_vec_r,
                                      const MeshInfo3D &meshInfo){

  CHECK_INPUT(primalVariable);     
  return cuda_TVL1SymOF3D_proxPrimal( primalVariable, primalStepSize_tau, primalFctWeight_Matching, rho_const_l, rho_vec_l, rho_const_r, rho_vec_r, meshInfo);                          
};


#endif