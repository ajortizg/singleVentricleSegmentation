#ifndef __COREDEFINES_H_
#define __COREDEFINES_H_

#include <torch/extension.h>
#include <vector>

//=======================================
// C++ interface
//=======================================
#define CHECK_CUDA(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



class MeshInfo1D {

public:
  
  const int _NX;
  const float _LX;
  const float _hX;

  MeshInfo1D( const int NX, const float LX ) : 
  _NX(NX), 
  _LX(LX), 
  _hX(LX/(_NX - 1)) {}

};

class MeshInfo2D {

public:
  
  const int _NY,_NX;
  const float _LY,_LX;
  const float _hY,_hX;

  MeshInfo2D( const int NY, const int NX, const float LY, const float LX ) : 
  _NY(NY), _NX(NX), 
  _LY(LY), _LX(LX),
  _hY(LY/(_NY - 1)), _hX(LX/(_NX - 1)) {}

};

class MeshInfo3D {

public:
  
  const int _NZ,_NY,_NX;
  const float _LZ,_LY,_LX;
  const float _hZ,_hY,_hX;

  MeshInfo3D( const int NZ, const int NY, const int NX, const float LZ, const float LY, const float LX ) : 
   _NZ(NZ), _NY(NY), _NX(NX),
   _LZ(LZ), _LY(LY), _LX(LX),
   _hZ(LZ/(_NZ - 1)), _hY(LY/(_NY - 1)), _hX(LX/(_NX - 1)) {}

};



/** Interpolation types. */
typedef enum
{
  //INTERPOLATE_NEAREST,
  INTERPOLATE_LINEAR,
  INTERPOLATE_CUBIC_HERMITESPLINE,
  INTERPOLATE_CUBIC_BSPLINE
} InterpolationType;



#endif