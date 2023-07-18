#ifndef dev_eval_indxing_trifocal_2op1p_30_direct_param_homotopy_cuh_
#define dev_eval_indxing_trifocal_2op1p_30_direct_param_homotopy_cuh_
// ===============================================================================================
// Code Description: Device function for evaluating the parallel indexing of the Jacobians w.r.t. 
//                   the unknowns x (Hx), the variable t (Ht), and the homotopy itself
// the trifocal 2op1p 30x30 problem
//
// Major Modifications
//    Chiang-Heng Chien  22-10-03:   Edited from the first version 
//                                   (dev-eval-indexing-trifocal_2op1p_30.cuh)
//
// ===============================================================================================
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

    //> compute the parameter homotopy
    template<int N>
    __device__ __inline__ void
    compute_param_homotopy(
      const int tx, float t,
      magmaFloatComplex *s_param_homotopy,
      magmaFloatComplex *s_start_params,
      magmaFloatComplex *s_target_params
    ) 
    {
        //> 30 threads with 33 parameters
        //> floor(33/30) = 1
        //> mod(33/30) = 3
        s_param_homotopy[ tx ] = s_target_params[ tx ] * t + s_start_params[ tx ] * (1.0-t);

        if (tx < 3) {
          s_param_homotopy[ tx + N ] = s_target_params[ tx + N ] * t + s_start_params[ tx + N ] * (1.0-t);
        }
    }

    //> Jacobian \partial H / \partial x parallel evaluation
    template<int N, int max_terms, int max_parts, int max_terms_parts>
    __device__ __inline__ void
    eval_Jacobian_Hx(
        const int tx, float t,                //> thread id and t
        magmaFloatComplex r_cgesvA[N],        //> each row of the Jacobian matrix
        magmaFloatComplex *s_vars,            //> variables
        magmaFloatComplex *s_start_params,    //> start parameters
        magmaFloatComplex *s_target_params,   //> target parameters
        magmaFloatComplex *s_param_homotopy,  //> parameter homotopy
        const int* __restrict__ d_Hx_indices//, //> indices for the Jacobian Hx matrix
    )
    {
      //> Full, explicit form of evaluation
      #pragma unroll
      for(int i = 0; i < N; i++) {

        //> initialize to zero
        r_cgesvA[i] = MAGMA_C_ZERO;

        //> Should I do transpose?????
        //> Without transpose...
        /*#pragma unroll
        for(int j = 0; j < max_terms; j++) {

          //> access the indices of parameters (should be access in a coalesced fashion)
          p1_indx = d_Hx_indices[ i*max_terms*max_parts + j*max_parts + tx*N*max_terms*max_parts + 1 ];
          p2_indx = d_Hx_indices[ i*max_terms*max_parts + j*max_parts + tx*N*max_terms*max_parts + 2 ];

          //> access the indices of variables (should also be access in a coalesced fashion)
          v1_indx = d_Hx_indices[ i*max_terms*max_parts + j*max_parts + tx*N*max_terms*max_parts + 3 ];
          v2_indx = d_Hx_indices[ i*max_terms*max_parts + j*max_parts + tx*N*max_terms*max_parts + 4 ];


          //> compute the element of the Jacobian matrix Hx
          r_cgesvA[i] += d_Hx_indices[ i*max_terms*max_parts + j*max_parts + tx*N*max_terms*max_parts ]
                       * s_param_homotopy[ p1_indx ]
                       * s_param_homotopy[ p2_indx ]
                       * (s_vars[ v1_indx ])
                       * (s_vars[ v2_indx ]);
        }*/

        //> With transpose...
        #pragma unroll
        for(int j = 0; j < max_terms; j++) {

          //> access the indices of parameters (should be access in a coalesced fashion)
          //p1_indx = d_Hx_indices[ (i*max_terms*max_parts)*N + (j*max_parts+1)*N + tx ];
          //p2_indx = d_Hx_indices[ (i*max_terms*max_parts)*N + (j*max_parts+2)*N + tx ];

          //> access the indices of variables (should also be access in a coalesced fashion)
          //v1_indx = d_Hx_indices[ (i*max_terms*max_parts)*N + (j*max_parts+3)*N + tx ];
          //v2_indx = d_Hx_indices[ (i*max_terms*max_parts)*N + (j*max_parts+4)*N + tx ];

          /*//> compute the element of the Jacobian matrix Hx
          r_cgesvA[i] += d_Hx_indices[(i*max_terms*max_parts)*N + j*max_parts*N + tx]
                       * s_param_homotopy[ p1_indx ]
                       * s_param_homotopy[ p2_indx ]
                       * (s_vars[ v1_indx ])
                       * (s_vars[ v2_indx ]);*/

          //> compute the element of the Jacobian matrix Hx
          r_cgesvA[i] += d_Hx_indices[(i*max_terms_parts)*N + j*max_parts*N + tx]
                       * s_param_homotopy[ d_Hx_indices[ (i*max_terms_parts)*N + (j*max_parts+1)*N + tx ] ]
                       * s_param_homotopy[ d_Hx_indices[ (i*max_terms_parts)*N + (j*max_parts+2)*N + tx ] ]
                       * (s_vars[ d_Hx_indices[ (i*max_terms_parts)*N + (j*max_parts+3)*N + tx ] ])
                       * (s_vars[ d_Hx_indices[ (i*max_terms_parts)*N + (j*max_parts+4)*N + tx ] ]);
        }
      }
    }

    //> Jacobian \partial H / \partial t parallel evaluation
    template<int N, int max_terms, int max_parts>
    __device__ __inline__ void
    eval_Jacobian_Ht(
        const int tx, float t,                //> thread id and t
        magmaFloatComplex &r_cgesvB,          //> each row of the Jacobian matrix
        magmaFloatComplex *s_vars,            //> variables
        magmaFloatComplex *s_start_params,    //> start parameters
        magmaFloatComplex *s_target_params,   //> target parameters
        magmaFloatComplex *s_param_homotopy,  //> parameter homotopy
        const int* __restrict__ d_Ht_indices,  //> indices for the Jacobian Hx matrix
        //const int* d_Ht_indices,  //> indices for the Jacobian Hx matrix
        magmaFloatComplex *s_diffParams//,
        //unsigned p1_indx,
        //unsigned p2_indx,
        //unsigned v1_indx,
        //unsigned v2_indx,
        //unsigned v3_indx
    )
    {
      //> initialize each element to 0
      r_cgesvB = MAGMA_C_ZERO;
      
      /*#pragma unroll
      for (int i = 0; i < max_terms; i++) {

        //> access the indices of parameters (should be access in a coalesced fashion)
        p1_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 1 ];
        p2_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 2 ];

        //> access the indices of variables (should also be access in a coalesced fashion)
        v1_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 3 ];
        v2_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 4 ];
        v3_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 5 ];

       
        //> With transpose...
        r_cgesvB -= d_Ht_indices[i*max_parts + tx*max_terms*max_parts]
                  * (s_diffParams[p1_indx] * s_param_homotopy[ p2_indx ]
                   + s_diffParams[p2_indx] * s_param_homotopy[ p1_indx ])
                  * (s_vars[ v1_indx ])
                  * (s_vars[ v2_indx ])
                  * (s_vars[ v3_indx ]);
      }*/

      #pragma unroll
      for (int i = 0; i < max_terms; i++) {

        /*//> access the indices of parameters (should be access in a coalesced fashion)
        p1_indx = d_Ht_indices[ (i*max_parts+1)*N + tx ];
        p2_indx = d_Ht_indices[ (i*max_parts+2)*N + tx ];

        //> access the indices of variables (should also be access in a coalesced fashion)
        v1_indx = d_Ht_indices[ (i*max_parts+3)*N + tx ];
        v2_indx = d_Ht_indices[ (i*max_parts+4)*N + tx ];
        v3_indx = d_Ht_indices[ (i*max_parts+5)*N + tx ];

        //> With transpose...
        r_cgesvB -= d_Ht_indices[i*max_parts*N + tx]
                  * (s_diffParams[p1_indx] * s_param_homotopy[ p2_indx ]
                   + s_diffParams[p2_indx] * s_param_homotopy[ p1_indx ] )
                  * (s_vars[ v1_indx ])
                  * (s_vars[ v2_indx ])
                  * (s_vars[ v3_indx ]);*/

        //> With transpose...
        r_cgesvB -= d_Ht_indices[i*max_parts*N + tx]
                  * (s_diffParams[d_Ht_indices[ (i*max_parts+1)*N + tx ]] * s_param_homotopy[ d_Ht_indices[ (i*max_parts+2)*N + tx ] ]
                   + s_diffParams[d_Ht_indices[ (i*max_parts+2)*N + tx ]] * s_param_homotopy[ d_Ht_indices[ (i*max_parts+1)*N + tx ] ] )
                  * (s_vars[ d_Ht_indices[ (i*max_parts+3)*N + tx ] ])
                  * (s_vars[ d_Ht_indices[ (i*max_parts+4)*N + tx ] ])
                  * (s_vars[ d_Ht_indices[ (i*max_parts+5)*N + tx ] ]);
      }
    }

    //> parameter homotopy evaluation
    template<int N, int max_terms, int max_parts>
    __device__ __inline__ void
    eval_Parameter_Homotopy(
        const int tx, float t,                //> thread id and t
        magmaFloatComplex &r_cgesvB,          //> each row of the parameter homotopy
        magmaFloatComplex *s_vars,            //> variables
        magmaFloatComplex *s_start_params,    //> start parameters
        magmaFloatComplex *s_target_params,   //> target parameters
        magmaFloatComplex *s_param_homotopy,  //> parameter homotopy
        const int* __restrict__ d_Ht_indices//, //> indices for the Jacobian Ht matrix
        //const int* d_Ht_indices, //> indices for the Jacobian Ht matrix
        //unsigned p1_indx,
        //unsigned p2_indx,
        //unsigned v1_indx,
        //unsigned v2_indx,
        //unsigned v3_indx
    )
    {
      //> initialize each element to 0
      r_cgesvB = MAGMA_C_ZERO;
      
      /*#pragma unroll
      for (int i = 0; i < max_terms; i++) {

        //> access the indices of parameters (should be access in a coalesced fashion)
        p1_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 1 ];
        p2_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 2 ];

        //> access the indices of variables (should also be access in a coalesced fashion)
        v1_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 3 ];
        v2_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 4 ];
        v3_indx = d_Ht_indices[ i*max_parts + tx*max_terms*max_parts + 5 ];

        r_cgesvB += d_Ht_indices[i*max_parts + tx*max_terms*max_parts]
                  * s_param_homotopy[ p1_indx ]
                  * s_param_homotopy[ p2_indx ]
                  * (s_vars[ v1_indx ])
                  * (s_vars[ v2_indx ])
                  * (s_vars[ v3_indx ]);
      }*/

      #pragma unroll
      for (int i = 0; i < max_terms; i++) {

        //> access the indices of parameters (should be access in a coalesced fashion)
        //p1_indx = d_Ht_indices[ (i*max_parts+1)*N + tx ];
        //p2_indx = d_Ht_indices[ (i*max_parts+2)*N + tx ];

        //> access the indices of variables (should also be access in a coalesced fashion)
        //v1_indx = d_Ht_indices[ (i*max_parts+3)*N + tx ];
        //v2_indx = d_Ht_indices[ (i*max_parts+4)*N + tx ];
        //v3_indx = d_Ht_indices[ (i*max_parts+5)*N + tx ];

        /*r_cgesvB += d_Ht_indices[i*max_parts*N + tx]
                  * s_param_homotopy[ p1_indx ]
                  * s_param_homotopy[ p2_indx ]
                  * (s_vars[ v1_indx ])
                  * (s_vars[ v2_indx ])
                  * (s_vars[ v3_indx ]);*/
        
        r_cgesvB += d_Ht_indices[i*max_parts*N + tx]
                  * s_param_homotopy[ d_Ht_indices[ (i*max_parts+1)*N + tx ] ]
                  * s_param_homotopy[ d_Ht_indices[ (i*max_parts+2)*N + tx ] ]
                  * (s_vars[ d_Ht_indices[ (i*max_parts+3)*N + tx ] ])
                  * (s_vars[ d_Ht_indices[ (i*max_parts+4)*N + tx ] ])
                  * (s_vars[ d_Ht_indices[ (i*max_parts+5)*N + tx ] ]);
      }
    }
}

#endif
