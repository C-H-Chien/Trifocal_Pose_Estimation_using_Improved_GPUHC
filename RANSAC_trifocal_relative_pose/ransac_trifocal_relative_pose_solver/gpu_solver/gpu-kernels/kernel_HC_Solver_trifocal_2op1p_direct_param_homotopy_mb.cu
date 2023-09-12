#ifndef kernel_HC_Solver_trifocal_2op1p_30_direct_param_homotopy_mb_cu
#define kernel_HC_Solver_trifocal_2op1p_30_direct_param_homotopy_mb_cu
// ============================================================================
// GPU homotopy continuation solver for the trifocal 2op1p 30x30 problem
// Version 2: Direct evaluation of parameter homotopy. The parameter homotopy 
//            part of each polynomial is not expanded to an uni-variable 
//            polynomial. Rather, depending on the order of t, the parameter 
//            homotopy formulation is explicitly hard-coded such that we do not 
//            need to compute coefficients from parameters first then use 
//            index-based to evaluate the Jacobians. The required amount of data 
//            to be stored in a kernel in this method is reduced which expects 
//            to speedup over the first version.
//
// Major Modifications
//    Chiang-Heng Chien  22-10-03:   Edited from the first version 
//                                   (kernel_HC_Solver_trifocal_2op1p_30.cu)
//    Chiang-Heng Chien  23-07-18:   Run under a RANSAC scheme with multiple 
//                                   batches and multiple HC trackins per warp
//
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>

// cuda included
#include <cuda.h>
#include <cuda_runtime.h>

// magma
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

//> header
#include "magmaHC-kernels.h"

//> device functions
#include "../gpu-dev-functions/dev-eval-indxing-trifocal_2op1p_30_direct_param_homotopy.cuh"
#include "../gpu-dev-functions/dev-cgesv-batched-small.cuh"
#include "../gpu-dev-functions/dev-get-new-data.cuh"

#include "../../definitions.h"

namespace magmaHCWrapper {

  template<int N, int num_of_params, int max_steps, int max_corr_steps, int predSuccessCount, 
           int Hx_max_terms, int Hx_max_parts, int Hx_max_terms_parts, int Ht_max_terms, int Ht_max_parts,
           int batchCount, int NUMBER_OF_BATCHES_MULTIPLES, int NUMBER_OF_TRACKINGS_PER_WARP>
  __global__ void
  HC_solver_trifocal_2op1p_30_direct_param_homotopy_mb(
    magma_int_t ldda, 
    magmaFloatComplex** d_startSols_array,
    magmaFloatComplex** d_Track_array,
    magmaFloatComplex*  d_startParams,
    magmaFloatComplex*  d_targetParams,
    magmaFloatComplex** d_cgesvA_array,
    magmaFloatComplex** d_cgesvB_array,
    magmaFloatComplex*  d_diffParams,
    const magma_int_t* __restrict__ d_Hx_indices,
    const magma_int_t* __restrict__ d_Ht_indices,
    magmaFloatComplex*  d_path_converge_flag
  )
  {
    extern __shared__ magmaFloatComplex zdata[];
    const int tx = threadIdx.x;
    const int batchid = blockIdx.x ;

    //> define pointers to the arrays
    magmaFloatComplex* d_startSols    = d_startSols_array[batchid];
    magmaFloatComplex* d_cgesvA       = d_cgesvA_array[batchid];
    magmaFloatComplex* d_cgesvB       = d_cgesvB_array[batchid];

    //> declarations of registers
    magmaFloatComplex r_cgesvA[N] = {MAGMA_C_ZERO};
    magmaFloatComplex r_cgesvB    = MAGMA_C_ZERO;
    
    //> declarations of shared memories
    magmaFloatComplex *s_startParams        = (magmaFloatComplex*)(zdata);
    magmaFloatComplex *s_targetParams       = s_startParams + (num_of_params + 1);
    magmaFloatComplex *s_diffParams         = s_targetParams + (num_of_params + 1);
    magmaFloatComplex *s_param_homotopy     = s_diffParams + (num_of_params + 1);
    magmaFloatComplex *s_sols               = s_param_homotopy + (num_of_params + 1);
    magmaFloatComplex *s_track              = s_sols + (N+1);
    magmaFloatComplex *s_track_last_success = s_track + (N+1);
    magmaFloatComplex *sB                   = s_track_last_success + (N+1);
    magmaFloatComplex *sx                   = sB + N;
    float *dsx                              = (float*)(sx + N);
    float *s_sqrt_sols                      = dsx + N;
    float *s_sqrt_corr                      = s_sqrt_sols + N;
    float *s_norm                           = s_sqrt_corr + N;
    int   *sipiv                            = (int*)(s_norm + 2);
    bool   s_isSuccessful                   = (bool)(sipiv + N);
    int    s_pred_success_count             = (int)(s_isSuccessful + 1);

    //> read data from global memory to registers
    #pragma unroll
    for(int i = 0; i < N; i++) {
      r_cgesvA[i] = d_cgesvA[ i * ldda + tx ];
    }
    r_cgesvB = d_cgesvB[tx];

    //> start and target parameters
    s_startParams[tx]  = d_startParams[tx];
    
    if (tx == 0) {
      //> the rest of the start and target parameters
      #pragma unroll
      for(int i = N; i <= num_of_params; i++) {
        s_startParams[i]  = d_startParams[i];
        //s_targetParams[i] = d_targetParams[i];
        //s_diffParams[i]   = d_diffParams[i];
      }
      s_sols[N]                       = MAGMA_C_MAKE(1.0, 0.0);
      s_track[N]                      = MAGMA_C_MAKE(1.0, 0.0);
      s_track_last_success[N]         = MAGMA_C_MAKE(1.0, 0.0);
      s_param_homotopy[num_of_params] = MAGMA_C_ONE;
    }
    __syncthreads();

    //> 1/2 \Delta t
    float one_half_delta_t;

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    int batch_mul_id = batchid / batchCount;
    for (int ri = 0; ri < NUMBER_OF_TRACKINGS_PER_WARP; ri++) {
      magmaFloatComplex* d_track = d_Track_array[batchid + ri*batchCount*NUMBER_OF_BATCHES_MULTIPLES];
      s_track[tx]                = d_track[tx];

      
      s_targetParams[tx]         = d_targetParams[tx + batch_mul_id * (num_of_params+1) + ri * NUMBER_OF_BATCHES_MULTIPLES * (num_of_params+1)];
      s_diffParams[tx]           = d_diffParams[tx + batch_mul_id * (num_of_params+1) + ri * NUMBER_OF_BATCHES_MULTIPLES * (num_of_params+1)];

      s_sols[tx]               = d_startSols[tx];
      s_track_last_success[tx] = s_track[tx];
      s_sqrt_sols[tx]          = 0;
      s_sqrt_corr[tx]          = 0;
      s_isSuccessful           = 0;
      s_pred_success_count     = 0;
      __syncthreads();

      int linfo = 0, rowid = tx;
      float t0 = 0.0, t_step = 0.0, delta_t = 0.01;
      bool end_zone = 0;
      int hc_step = 0;

      #pragma unroll
      for(int i = N; i <= num_of_params; i++) {
        s_targetParams[i] = d_targetParams[i];
        s_diffParams[i]   = d_diffParams[i];
      }

      //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      /*if (tx == 0 && batchid == 1) {
        printf("Round #%d\n", ri);
        for (int ii = 0; ii < 31; ii++) {
          printf("%.5f\t%.5f\n", MAGMA_C_REAL(s_track[ii]), MAGMA_C_IMAG(s_track[ii]));
        }
        printf("\n");
      }

      if (tx == 0 && batchid == 1) {
        printf("Round #%d\n", ri);
        for (int ii = 0; ii < 34; ii++) {
          printf("%.5f\t%.5f\n", MAGMA_C_REAL(s_targetParams[ii]), MAGMA_C_IMAG(s_targetParams[ii]));
        }
        printf("\n");
      }*/

      //#pragma unroll
      for (int step = 0; step <= max_steps; step++) {
        if (t0 < 1.0 && (1.0-t0 > 0.0000001)) {

          // ===================================================================
          // Decide delta t at end zone
          // ===================================================================
          if (!end_zone && fabs(1 - t0) <= (0.0500001)) {
            end_zone = true;

            //> TEST!!!!!!!!!!!!!!!!!!!!!
            //break;
          }

          if (end_zone) {
            if (delta_t > fabs(1 - t0))
              delta_t = fabs(1 - t0);
          }
          else if (delta_t > fabs(0.95 - t0)) {
            delta_t = fabs(0.95 - t0);
          }

          t_step = t0;
          one_half_delta_t = 0.5 * delta_t;
          // ===================================================================
          // Prediction: 4-th order Runge-Kutta method
          // ===================================================================
          //>  get HxHt for k1
          compute_param_homotopy<N>( tx, t0, s_param_homotopy, s_startParams, s_targetParams );
          eval_Jacobian_Hx<N, Hx_max_terms, Hx_max_parts, Hx_max_terms_parts>( tx, t0, r_cgesvA, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Hx_indices );
          eval_Jacobian_Ht<N, Ht_max_terms, Ht_max_parts>( tx, t0, r_cgesvB, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Ht_indices, s_diffParams );

          // -- solve k1 --
          cgesv_batched_small_device<N>( tx, r_cgesvA, sipiv, r_cgesvB, sB, sx, dsx, rowid, linfo );
          magmablas_syncwarp();

          // -- compute x for the creation of HxHt for k2 --
          create_x_for_k2( tx, t0, delta_t, one_half_delta_t, s_sols, s_track, sB );
          magmablas_syncwarp();

          // -- get HxHt for k2 --
          compute_param_homotopy<N>( tx, t0, s_param_homotopy, s_startParams, s_targetParams );
          eval_Jacobian_Hx<N, Hx_max_terms, Hx_max_parts, Hx_max_terms_parts>( tx, t0, r_cgesvA, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Hx_indices );
          eval_Jacobian_Ht<N, Ht_max_terms, Ht_max_parts>( tx, t0, r_cgesvB, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Ht_indices, s_diffParams );

          // -- solve k2 --
          cgesv_batched_small_device<N>( tx, r_cgesvA, sipiv, r_cgesvB, sB, sx, dsx, rowid, linfo );
          magmablas_syncwarp();

          // -- compute x for the generation of HxHt for k3 --
          create_x_for_k3( tx, delta_t, one_half_delta_t, s_sols, s_track, s_track_last_success, sB );
          magmablas_syncwarp();

          // -- get HxHt for k3 --
          //compute_param_homotopy<N>( tx, t0, s_param_homotopy, s_start_params, s_target_params );
          eval_Jacobian_Hx<N, Hx_max_terms, Hx_max_parts, Hx_max_terms_parts>( tx, t0, r_cgesvA, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Hx_indices );
          eval_Jacobian_Ht<N, Ht_max_terms, Ht_max_parts>( tx, t0, r_cgesvB, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Ht_indices, s_diffParams );

          // -- solve k3 --
          cgesv_batched_small_device<N>( tx, r_cgesvA, sipiv, r_cgesvB, sB, sx, dsx, rowid, linfo );
          magmablas_syncwarp();

          // -- compute x for the generation of HxHt for k4 --
          create_x_for_k4( tx, t0, delta_t, one_half_delta_t, s_sols, s_track, s_track_last_success, sB );
          magmablas_syncwarp();

          // -- get HxHt for k4 --
          compute_param_homotopy<N>( tx, t0, s_param_homotopy, s_startParams, s_targetParams );
          eval_Jacobian_Hx<N, Hx_max_terms, Hx_max_parts, Hx_max_terms_parts>( tx, t0, r_cgesvA, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Hx_indices );
          eval_Jacobian_Ht<N, Ht_max_terms, Ht_max_parts>( tx, t0, r_cgesvB, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Ht_indices, s_diffParams );

          // -- solve k4 --
          cgesv_batched_small_device<N>( tx, r_cgesvA, sipiv, r_cgesvB, sB, sx, dsx, rowid, linfo );
          magmablas_syncwarp();

          // -- make prediction --
          s_sols[tx] += sB[tx] * delta_t * 1.0/6.0;
          s_track[tx] = s_sols[tx];
          __syncthreads();

          // ===================================================================
          // -- Gauss-Newton Corrector --
          // ===================================================================
          //#pragma unroll
          for(int i = 0; i < max_corr_steps; i++) {

            //> evaluate the Jacobian Hx
            eval_Jacobian_Hx<N, Hx_max_terms, Hx_max_parts, Hx_max_terms_parts>( tx, t0, r_cgesvA, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Hx_indices );

            //> evaluate the parameter homotopy
            eval_Parameter_Homotopy<N, Ht_max_terms, Ht_max_parts>( tx, t0, r_cgesvB, s_track, s_startParams, s_targetParams, s_param_homotopy, d_Ht_indices );

            //> G-N corrector first solve
            cgesv_batched_small_device<N>( tx, r_cgesvA, sipiv, r_cgesvB, sB, sx, dsx, rowid, linfo );
            magmablas_syncwarp();

            //> correct the sols
            s_track[tx] -= sB[tx];
            __syncthreads();

            //> compute the norms; norm[0] is norm(sB), norm[1] is norm(sol)
            compute_norm2<N>( tx, sB, s_track, s_sqrt_sols, s_sqrt_corr, s_norm );
            __syncthreads();
            
            s_isSuccessful = s_norm[0] < 0.000001 * s_norm[1];
            __syncthreads();

            if (s_isSuccessful)
              break;
          }

          //> stop if the values of the solution is too large
          if ((s_norm[1] > 1e14) && (t0 < 1.0) && (1.0-t0 > 0.001)) {
            //inf_failed = 1;
            break;
          }

          // ===================================================================
          // Decide Track Changes
          // ===================================================================
          if (!s_isSuccessful) {
            s_pred_success_count = 0;
            delta_t *= 0.5;
            //> should be the last successful tracked sols
            s_track[tx] = s_track_last_success[tx];
            s_sols[tx] = s_track_last_success[tx];
            __syncthreads();
            t0 = t_step;
          }
          else {
            s_track_last_success[tx] = s_track[tx];
            s_sols[tx] = s_track[tx];
            __syncthreads();
            s_pred_success_count++;
            if (s_pred_success_count >= predSuccessCount) {
              s_pred_success_count = 0;
              delta_t *= 2;
            }
          }
          hc_step++;
        }
        else {
          break;
        }
      }

      //> d_cgesvB tells whether the track is finished, if not, stores t0 and delta_t
      d_path_converge_flag[batchid + ri*batchCount] = (t0 >= 1.0 || (1.0-t0 <= 0.0000001)) ? MAGMA_C_MAKE(1.0, hc_step) : MAGMA_C_MAKE(t0, delta_t);

      //> d_track stores the solutions
      d_track[tx] = s_track[tx];
    }
  }

  extern "C" real_Double_t
  kernel_HC_Solver_trifocal_2op1p_30_direct_param_homotopy_mb(
    magma_queue_t my_queue,
    magma_int_t ldda,
    magma_int_t N, 
    magma_int_t num_of_params,
    magma_int_t batchCount, 
    magmaFloatComplex** d_startSols_array, 
    magmaFloatComplex** d_Track_array, //
    magmaFloatComplex*  d_startParams,
    magmaFloatComplex*  d_targetParams, //
    magmaFloatComplex** d_cgesvA_array, 
    magmaFloatComplex** d_cgesvB_array,
    magmaFloatComplex*  d_diffParams, //
    magma_int_t* d_Hx_indx, 
    magma_int_t* d_Ht_indx,
    magmaFloatComplex*  d_path_converge_flag)
  {
    real_Double_t gpu_time;
    const magma_int_t thread_x = N;
    dim3 threads(thread_x, 1, 1);
    dim3 grid(batchCount, 1, 1);
    cudaError_t e = cudaErrorInvalidValue;

    //std::cout << "batchCount = " << batchCount << std::endl;

    //> declare the amount of shared memory for the use of the kernel
    magma_int_t shmem  = 0;
    shmem += (num_of_params+1) * sizeof(magmaFloatComplex);  //> start parameters
    shmem += (num_of_params+1) * sizeof(magmaFloatComplex);  //> target parameters
    shmem += (num_of_params+1) * sizeof(magmaFloatComplex);  //> difference of start and target parameters
    shmem += (num_of_params+1) * sizeof(magmaFloatComplex);  //> parameter homotopy used when t is changed
    shmem += (N+1) * sizeof(magmaFloatComplex);              //> start solutions
    shmem += (N+1) * sizeof(magmaFloatComplex);              //> intermediate solutions
    shmem += (N+1) * sizeof(magmaFloatComplex);              //> last successful intermediate solutions
    shmem += N * sizeof(magmaFloatComplex);                  //> linear system solution
    shmem += N * sizeof(magmaFloatComplex);                  //> intermediate varaible for cgesv
    shmem += N * sizeof(float);                              //> intermediate varaible for cgesv
    shmem += N * sizeof(int);                                //> squared solution
    shmem += N * sizeof(float);                              //> squared correction solution
    shmem += N * sizeof(float);                              //> solution norm
    shmem += 2 * sizeof(float);                              //> pivot for cgesv
    shmem += 1 * sizeof(bool);                               //> is_successful 
    shmem += 1 * sizeof(int);                                //> predictor successes counter

    //> declare kernel arguments  
    void *kernel_args[] = {&ldda, 
                           &d_startSols_array,
                           &d_Track_array,
                           &d_startParams, &d_targetParams,
                           &d_cgesvA_array, &d_cgesvB_array,
                           &d_diffParams,
                           &d_Hx_indx, &d_Ht_indx,
                           &d_path_converge_flag
                          };

    gpu_time = magma_sync_wtime( my_queue );

    //int batchCount, int NUMBER_OF_BATCHES_MULTIPLES, int NUMBER_OF_TRACKINGS_PER_WARP>

    //> launch the GPU kernel
    //> < Number of Unknowns, Number of Parameters, Maximal Steps, Number of correction steps, Number of steps to be successful, Don't care...>
    //> LAST THREE ARGUMENTS: (int batchCount, int NUMBER_OF_BATCHES_MULTIPLES, int NUMBER_OF_TRACKINGS_PER_WARP)
    e = cudaLaunchKernel((void*)HC_solver_trifocal_2op1p_30_direct_param_homotopy_mb
                         < 30, 33, 100, 5, 10, 8, 5, 40, 16, 6, 312, MULTIPLES_OF_BATCHCOUNT, MULTIPLES_OF_TRACKING_PER_WARP >, 
                         grid, threads, kernel_args, shmem, my_queue->cuda_stream());


    gpu_time = magma_sync_wtime( my_queue ) - gpu_time;
    if( e != cudaSuccess ) {
        printf("cudaLaunchKernel of HC_solver_trifocal_2op1p_30_direct_param_homotopy is not successful!\n");
    }

    return gpu_time;
  }

}

#endif
