#ifndef dev_eval_indxing_trifocal_2op1p_30x30_P2C_cuh_
#define dev_eval_indxing_trifocal_2op1p_30x30_P2C_cuh_
// ===============================================================================================
// Code Description: Device function for evaluating the parallel indexing of the Jacobians w.r.t. 
//                   the unknowns x (Hx), the variable t (Ht), and the homotopy itself
//
// Major Modifications
//    Chien  24-03-24:   Edited from the dev-eval-indxing-trifocal_2op1p_30x30.cuh
//
// ===============================================================================================
#include <cstdio>
#include <cuda_runtime.h>

#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_internal.h"
#undef max
#undef min
#include "magma_templates.h"
#include "sync.cuh"
#undef max
#undef min
#include "shuffle.cuh"
#undef max
#undef min
#include "batched_kernel_param.h"

//> Macros
#include "../definitions.hpp"

//> Compute the coefficients from linear interpolations of parameters of phc
template< int Num_Of_Vars, int Max_Order_of_t, unsigned Full_Parallel_Offset, \
          unsigned Partial_Parallel_Thread_Offset, unsigned Partial_Parallel_Index_Offset, \
          unsigned Max_Order_of_t_Plus_One, unsigned Partial_Parallel_Index_Offset_Hx, unsigned Partial_Parallel_Index_Offset_Ht >
__device__ __noinline__ void
eval_parameter_homotopy(
    const int tx, float t, 
    magmaFloatComplex *s_phc_coeffs_Hx,
    magmaFloatComplex *s_phc_coeffs_Ht,
    const magmaFloatComplex __restrict__ *d_phc_coeffs_Hx,
    const magmaFloatComplex __restrict__ *d_phc_coeffs_Ht )
{
    // =============================================================================
    //> parameter homotopy for evaluating ∂H/∂x and ∂H/∂t
    for (int i = 0; i < Full_Parallel_Offset; i++) {
      s_phc_coeffs_Hx[ tx + i*Num_Of_Vars ] = d_phc_coeffs_Hx[ tx*Max_Order_of_t_Plus_One + i*Num_Of_Vars*Max_Order_of_t_Plus_One ] 
                                            + d_phc_coeffs_Hx[ tx*Max_Order_of_t_Plus_One + 1 + i*Num_Of_Vars*Max_Order_of_t_Plus_One ] * t
                                            + (d_phc_coeffs_Hx[ tx*Max_Order_of_t_Plus_One + 2 + i*Num_Of_Vars*Max_Order_of_t_Plus_One ] * t) * t;

      s_phc_coeffs_Ht[ tx + i*Num_Of_Vars ] = d_phc_coeffs_Ht[ tx*Max_Order_of_t + i*Num_Of_Vars*Max_Order_of_t ] 
                                            + d_phc_coeffs_Ht[ tx*Max_Order_of_t + 1 + i*Num_Of_Vars*Max_Order_of_t ] * t;
    }

    //> the remaining part
    if (tx < Partial_Parallel_Thread_Offset) {
      s_phc_coeffs_Hx[ tx + Partial_Parallel_Index_Offset ] = \
          d_phc_coeffs_Hx[ tx*Max_Order_of_t_Plus_One + Partial_Parallel_Index_Offset_Hx ] 
        + d_phc_coeffs_Hx[ tx*Max_Order_of_t_Plus_One + Partial_Parallel_Index_Offset_Hx + 1 ] * t
        + (d_phc_coeffs_Hx[ tx*Max_Order_of_t_Plus_One + Partial_Parallel_Index_Offset_Hx + 2 ] * t) * t;

      s_phc_coeffs_Ht[ tx + Partial_Parallel_Index_Offset ] = \
          d_phc_coeffs_Ht[ tx*Max_Order_of_t + Partial_Parallel_Index_Offset_Ht ] 
        + d_phc_coeffs_Ht[ tx*Max_Order_of_t + Partial_Parallel_Index_Offset_Ht + 1 ] * t;
    }
}

//> Parallel Jacobian Evaluation ∂H/∂x
template< int Num_Of_Vars, int dHdx_Max_Terms, int dHdx_Max_Parts, int dHdx_Entry_Offset, int dHdx_Row_Offset >
__device__ __noinline__ void
eval_Jacobian_Hx(
    const int tx, magmaFloatComplex *s_track, magmaFloatComplex r_cgesvA[Num_Of_Vars],
    const int* __restrict__ d_Hx_idx, magmaFloatComplex *s_phc_coeffs )
{
  for(int i = 0; i < Num_Of_Vars; i++) {
    r_cgesvA[i] = MAGMA_C_ZERO;

    for(int j = 0; j < dHdx_Max_Terms; j++) {
      r_cgesvA[i] += d_Hx_idx[j*dHdx_Max_Parts + i*dHdx_Entry_Offset + tx*dHdx_Row_Offset] 
                   * s_phc_coeffs[ d_Hx_idx[j*dHdx_Max_Parts + 1 + i*dHdx_Entry_Offset + tx*dHdx_Row_Offset] ]
                   * s_track[      d_Hx_idx[j*dHdx_Max_Parts + 2 + i*dHdx_Entry_Offset + tx*dHdx_Row_Offset] ]
                   * s_track[      d_Hx_idx[j*dHdx_Max_Parts + 3 + i*dHdx_Entry_Offset + tx*dHdx_Row_Offset] ] ;
    }
  }
}

//> Parallel Jacobian Evaluation ∂H/∂t
template< int dHdt_Max_Terms, int dHdt_Max_Parts, int dHdt_Row_Offset >
__device__ __noinline__ void
eval_Jacobian_Ht(
    const int tx, magmaFloatComplex *s_track, magmaFloatComplex &r_cgesvB,
    const int* __restrict__ d_Ht_idx, magmaFloatComplex *s_phc_coeffs)
{
  r_cgesvB = MAGMA_C_ZERO;
  for (int i = 0; i < dHdt_Max_Terms; i++) {
    r_cgesvB -= d_Ht_idx[ i*dHdt_Max_Parts + tx*dHdt_Row_Offset ] 
              * s_phc_coeffs[ d_Ht_idx[i*dHdt_Max_Parts + 1 + tx*dHdt_Row_Offset] ]
              * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 2 + tx*dHdt_Row_Offset] ] 
              * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 3 + tx*dHdt_Row_Offset] ] 
              * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 4 + tx*dHdt_Row_Offset] ];
  }
}

//> Parallel Homotopy Evaluation
template< int dHdt_Max_Terms, int dHdt_Max_Parts, int dHdt_Row_Offset >
__device__ __noinline__ void
eval_Homotopy(
    const int tx, magmaFloatComplex *s_track, magmaFloatComplex &r_cgesvB,
    const int* __restrict__ d_Ht_idx, magmaFloatComplex *s_phc_coeffs)
{
  r_cgesvB = MAGMA_C_ZERO;
  for (int i = 0; i < dHdt_Max_Terms; i++) {
    r_cgesvB += d_Ht_idx[ i*dHdt_Max_Parts + tx*dHdt_Row_Offset ] 
              * s_phc_coeffs[ d_Ht_idx[i*dHdt_Max_Parts + 1 + tx*dHdt_Row_Offset] ]
              * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 2 + tx*dHdt_Row_Offset] ] 
              * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 3 + tx*dHdt_Row_Offset] ] 
              * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 4 + tx*dHdt_Row_Offset] ] ;
  }
}

#endif