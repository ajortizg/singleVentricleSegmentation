



//============================================================================
// gradient operator
//============================================================================
/**
  * perform bilinear interpolation adjoint
  */
// template <typename T>
// __device__ T backpolate_bilinear(torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_x,
//                                  T& grad_idx, T& grad_idy,
//                                  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u, 
//                                  const int NY, const int NX,
//                                  T val, T idy, T idx) {
//   const int idx_f = floorf(idx);
//   const int idy_f = floorf(idy);

//   const int idx_c = idx_f + 1;
//   const int idy_c = idy_f + 1;

//   const T w = idx - idx_f;
//   const T h = idy - idy_f;

//   T i_ff = 0, i_fc = 0;
//   if (idx_f >= 0 && idx_f < grad_x.dimensions()[4]) {
//     if (idy_f >= 0 && idy_f < grad_x.dimensions()[3]) {
//       //tficg::CudaAtomicAdd(&grad_x(idy_f, idx_f),  (1 - h) * (1 - w) * val);
//       atomicAdd( &grad_x(idy_f, idx_f),  (1 - h) * (1 - w) * val);
//       i_ff = u[idy_f][idx_f];
//     }

//     if (idy_c >= 0 && idy_c < grad_x.dimensions()[3]) {
//       //tficg::CudaAtomicAdd(&grad_x(idy_c, idx_f),  h * (1 - w) * val);
//       atomicAdd( &grad_x(idy_c, idx_f),  h * (1 - w) * val) );
//       i_fc = u[idy_c][idx_f];
//     }
//   }

//   T i_cf = 0, i_cc = 0;
//   if (idx_c >= 0 && idx_c < grad_x.dimensions()[4]) {
//     if (idy_f >= 0 && idy_f < grad_x.dimensions()[3]) {
//       //tficg::CudaAtomicAdd(&grad_x(idy_f, idx_c), (1 - h) * w * val);
//       atomicAdd( &grad_x(idy_f, idx_c), (1 - h) * w * val );
//       i_cf = u[idy_f][idx_c];
//     }

//     if (idy_c >= 0 && idy_c < grad_x.dimensions()[3]) {
//       //tficg::CudaAtomicAdd(&grad_x(idy_c, idx_c), h * w * val);
//       atomicAdd( &grad_x(idy_c, idx_c), h * w * val );
//       i_cc = u[idy_c][idx_c];
//     }
//   }

//   grad_idx += ((1 - h) * (i_cf - i_ff) + h * (i_cc - i_fc)) * val;
//   grad_idy += ((1 - w) * (i_fc - i_ff) + w * (i_cc - i_cf)) * val;

//   T out = (1 - h) * (1 - w) * i_ff;
//   out += (1 - h) * w * i_cf;
//   out += h * (1 - w) * i_fc;
//   out += h * w * i_cc;

//   return out;
// }





template <typename T>
__device__ T backpolate_bilinear(
    const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u, 
    const int NY, const int NX,
    const float LY, const float LX,
    const float hY, const float hX,
    const int boundary,
    T val, T idy, T idx,
    torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_u,
    T& grad_idy, T& grad_idx) {
  
  const int ix_f = floorf(coord_x_warped / hX );
  const int ix_c = ix_f + 1;
  const T wx = coord_x_warped / hX - ix_f;
  const int ix_f_out = getIndexInterpolate(ix_f,NX,boundary);  
  const int ix_c_out = getIndexInterpolate(ix_c,NX,boundary);

  const int iy_f = floorf(coord_y_warped / hY );
  const int iy_c = iy_f + 1;
  const T wy = coord_y_warped / hY - iy_f;
  const int iy_f_out = getIndexInterpolate(iy_f,NY,boundary);  
  const int iy_c_out = getIndexInterpolate(iy_c,NY,boundary);

  T u_ff = u[iy_f_out][ix_f_out];
  T u_fc = u[iy_c_out][ix_f_out];
  T u_cf = u[iy_f_out][ix_c_out];
  T u_cc = u[iy_c_out][ix_c_out];

  T out = (1 - wy) * (1 - wx) * u_ff;
  out += (1 - wy) * wx * u_cf;
  out += wy * (1 - wx) * u_fc;
  out += wy * wx * u_cc;

  atomicAdd( &grad_u(iy_f_out, ix_f_out),  (1 - wy) * (1 - wx) * val);
  atomicAdd( &grad_u(iy_c_out, ix_f_out),  wy * (1 - wx) * val) );
  atomicAdd( &grad_u(iy_f_out, ix_c_out), (1 - wy) * wx * val );
  atomicAdd( &grad_u(iy_c_out, ix_c_out), wy * wx * val );

  // gradient in flow
  // grad_idx += ((1 - wy) * (u_cf - u_ff) + wy * (u_cc - u_fc)) * val;
  // grad_idy += ((1 - wx) * (u_fc - u_ff) + wx * (u_cc - u_cf)) * val;

  return out;
}








/**
 * perform cubic interpolation on the input image in given the index (idx,idy)
 */
template<typename T>
__device__ void backpolate_cubic_local(
     volatile T *grad_x, 
     T& grad_idx,
     T* in, T error, 
     T idx, 
     int kernel_size)
{
  const int idx_f = floorf(idx);
  const int idx_f_1 = idx_f - 1;
  const int idx_c = idx_f+1;
  const int idx_c_1 = idx_c+1;

  const T u = idx - idx_f;
  const T uu = u*u;
  const T uuu = uu*u;

  // determine the coefficients
  T d_out_d_p_f_1 = -uuu/2 + uu - u/2;
  T d_out_d_p_f = (3*uuu)/2 - (5*uu)/2 + 1;
  T d_out_d_p_c = -(3*uuu)/2 + 2*uu + u/2;
  T d_out_d_p_c_1 = uuu/2 - uu/2;

  T i_f = 0;
  if (idx_f >= 0 && idx_f < kernel_size)
  {
    i_f = in[idx_f];
    grad_x[idx_f] = d_out_d_p_f   * error;
  }
  else
    grad_x[idx_f] = 0;

  T i_f_1 = 0;
  if (idx_f_1 >= 0 && idx_f_1 < kernel_size)
  {
    i_f_1 = in[idx_f_1];
    grad_x[idx_f_1] = d_out_d_p_f_1 * error;
  }
  else
    grad_x[idx_f_1] = 0;

  T i_c = 0;
  if (idx_c >= 0 && idx_c < kernel_size)
  {
    i_c = in[idx_c];
    grad_x[idx_c] = d_out_d_p_c   * error;
  }
  else
    grad_x[idx_c] = 0;

  T i_c_1 = 0;
  if (idx_c_1 >= 0 && idx_c_1 < kernel_size)
  {
    i_c_1 = in[idx_c_1];
    grad_x[idx_c_1] = d_out_d_p_c_1 * error;
  }
  else
    grad_x[idx_c_1] = 0;

  // determine the coefficients
  const T p_f = i_f;
  const T p_prime_f = (i_c - i_f_1) / 2;
  const T p_c = i_c;
  const T p_prime_c = (i_c_1 - i_f) / 2;

  const T a = 2 * p_f - 2 * p_c + p_prime_f + p_prime_c;
  const T b = -3 * p_f + 3 * p_c - 2 * p_prime_f - p_prime_c;
  const T c = p_prime_f;

  grad_idx += (3*uu*a + 2*b*u + c) * error;
}



template<typename T>
__device__ T backpolate_bicubic(torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_x,
                                T& grad_idx, T& grad_idy,
                                const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> u,
                                const int NY, const int NX,
                                 T val, T idy, T idx) )
{
  const int idy_f = floorf(idy);
  const int idx_f = floorf(idx);

  T buff_y[4];
  T buff_x[4];

  // first perform interpolation
  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_idx_y = idy_f + dy;

    if (c_idx_y >= 0 && c_idx_y < x.dimensions()[3])
    {
      // get the input values
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
          buff_x[dx + 1] = u[c_idx_y][c_idx_x];
        else
          buff_x[dx + 1] = 0;
      }
      buff_y[dy + 1] = interpolate_cubic<T>(buff_x, idx - idx_f + 1, 4);
    }
    else
      buff_y[dy + 1] = 0;
  }

  T out = interpolate_cubic<T>(buff_y, idy - idy_f + 1, 4);

  // backpolate the error
  T buff_grad_y[4];
  cuda_interpolate1d_cubicHermiteSpline_backward_local<T>(buff_grad_y, grad_idy, buff_y, val, idy - idy_f + 1, 4);

  T buff_grad_x[4];
  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_idx_y = idy_f + dy;

    if (c_idx_y >= 0 && c_idx_y < x.dimensions()[3])
    {
      // get the input values
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
          buff_x[dx + 1] = u[c_idx_y][c_idx_x];
        else
          buff_x[dx + 1] = 0;
      }
      cuda_interpolate1d_cubicHermiteSpline_backward_local<T>(buff_grad_x, grad_idx, buff_x, buff_grad_y[dy+1], idx - idx_f + 1, 4);
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
          tficg::CudaAtomicAdd(&grad_x(c_idx_y, c_idx_x), buff_grad_x[dx + 1]);
      }
    }
  }
  return out;
}