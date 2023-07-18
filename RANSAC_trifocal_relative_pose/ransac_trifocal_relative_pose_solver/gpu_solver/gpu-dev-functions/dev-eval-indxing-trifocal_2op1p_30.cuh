#ifndef dev_eval_indxing_trifocal_2op1p_30_cuh_
#define dev_eval_indxing_trifocal_2op1p_30_cuh_
// ========================================================================================
// Device function for evaluating the parallel indexing for the Jacobians Hx, Ht, 
// and the homotopy H of the trifocal_2op1p_30 problem
//
// Modifications
//    Chiang-Heng Chien  23-06-19:   Initially created. Copied and edited from the 
//                                   dev-eval-indxing-six_lines_16.cuh file.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ========================================================================================
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>

// -- cuda included --
#include <cuda_runtime.h>

// -- magma included --
#include "flops.h"
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

namespace magmaHCWrapper {

    //> compute the linear interpolations of parameters of phc
    template<int N, int N_2, int N_3>
    __device__ __inline__ void
    eval_parameter_homotopy(
        const int tx, float t, 
        magmaFloatComplex *s_phc_coeffs_Hx,
        magmaFloatComplex *s_phc_coeffs_Ht,
        const magmaFloatComplex __restrict__ *d_phc_coeffs_Hx,
        const magmaFloatComplex __restrict__ *d_phc_coeffs_Ht
    )
    {
        // =============================================================================
        //> parameter homotopy of Hx
        //> floor(38/30) = 1
        #pragma unroll
        for (int i = 0; i < 1; i++) {
          s_phc_coeffs_Hx[ tx + i*N ] = d_phc_coeffs_Hx[ tx*3 + i*N_3 ] 
                                      + d_phc_coeffs_Hx[ tx*3 + 1 + i*N_3 ] * t
                                      + (d_phc_coeffs_Hx[ tx*3 + 2 + i*N_3 ] * t) * t;
          s_phc_coeffs_Ht[ tx + i*N ] = d_phc_coeffs_Ht[ tx*2 + i*N_2] 
                                      + d_phc_coeffs_Ht[ tx*2 + 1 + i*N_2 ] * t;
        }

        //> The remaining parts
        //> 38 - 30*1 = 38 - 30 = 8
        //> 37 - 30*1 = 37 - 30 = 7
        if (tx < 7) {
          //> parameter homotopy of Hx
          s_phc_coeffs_Hx[ tx + 30 ] = d_phc_coeffs_Hx[ tx*3 + 90 ] 
                                     + d_phc_coeffs_Hx[ tx*3 + 1 + 90 ] * t
                                     + (d_phc_coeffs_Hx[ tx*3 + 2 + 90 ] * t) * t;
          
          //> parameter homotopy of Ht
          s_phc_coeffs_Ht[ tx + 30 ] = d_phc_coeffs_Ht[ tx*2 + 60 ] 
                                     + d_phc_coeffs_Ht[ tx*2 + 1 + 60 ] * t;
        }
    }

    // -- Hx parallel indexing --
    template<int N, int max_terms, int max_parts, int max_terms_parts, int N_max_terms_parts>
    __device__ __inline__ void
    eval_Jacobian_Hx(
        const int tx, magmaFloatComplex *s_track, magmaFloatComplex r_cgesvA[N],
        const int* __restrict__ d_Hx_idx, magmaFloatComplex *s_phc_coeffs )
    {
      //#pragma unroll
      for(int i = 0; i < N; i++) {
        r_cgesvA[i] = MAGMA_C_ZERO;

        //#pragma unroll
        for(int j = 0; j < max_terms; j++) {
          r_cgesvA[i] += d_Hx_idx[j*max_parts + i*max_terms_parts + tx*N_max_terms_parts] 
                       * s_phc_coeffs[ d_Hx_idx[j*max_parts + 1 + i*max_terms_parts + tx*N_max_terms_parts] ]
                       * s_track[ d_Hx_idx[j*max_parts + 2 + i*max_terms_parts + tx*N_max_terms_parts] ]
                       * s_track[ d_Hx_idx[j*max_parts + 3 + i*max_terms_parts + tx*N_max_terms_parts] ];
        }
      }
    }

    // -- Ht parallel indexing --
    template<int N, int max_terms, int max_parts, int max_terms_parts>
    __device__ __inline__ void
    eval_Jacobian_Ht(
        const int tx, magmaFloatComplex *s_track, magmaFloatComplex &r_cgesvB,
        const int* __restrict__ d_Ht_idx, magmaFloatComplex *s_phc_coeffs)
    {
      r_cgesvB = MAGMA_C_ZERO;
      //#pragma unroll
      for (int i = 0; i < max_terms; i++) {
        r_cgesvB -= d_Ht_idx[i*max_parts + tx*max_terms_parts] 
                  * s_phc_coeffs[ d_Ht_idx[i*max_parts + 1 + tx*max_terms_parts] ]
                  * s_track[ d_Ht_idx[i*max_parts + 2 + tx*max_terms_parts] ] 
                  * s_track[ d_Ht_idx[i*max_parts + 3 + tx*max_terms_parts] ]
                  * s_track[ d_Ht_idx[i*max_parts + 4 + tx*max_terms_parts] ];
      }
    }

    // -- H parallel indexing --
    template<int N, int max_terms, int max_parts, int max_terms_parts>
    __device__ __inline__ void
    eval_homotopy(
        const int tx, magmaFloatComplex *s_track, magmaFloatComplex &r_cgesvB,
        const int* __restrict__ d_Ht_idx, magmaFloatComplex *s_phc_coeffs)
    {
      r_cgesvB = MAGMA_C_ZERO;
      //#pragma unroll
      for (int i = 0; i < max_terms; i++) {
        r_cgesvB += d_Ht_idx[i*max_parts + tx*max_terms_parts] 
                  * s_phc_coeffs[ d_Ht_idx[i*max_parts + 1 + tx*max_terms_parts] ]
                  * s_track[ d_Ht_idx[i*max_parts + 2 + tx*max_terms_parts] ] 
                  * s_track[ d_Ht_idx[i*max_parts + 3 + tx*max_terms_parts] ]
                  * s_track[ d_Ht_idx[i*max_parts + 4 + tx*max_terms_parts] ];
      }
    }
}

#endif
