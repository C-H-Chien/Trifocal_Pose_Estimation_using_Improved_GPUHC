#ifndef HC_TRACK_CHICAGO_PROBLEM_CPP
#define HC_TRACK_CHICAGO_PROBLEM_CPP
// ===================================================================================
//
// Modifications
//    Chiang-Heng Chien  23-07-04:   Initially shifted from previous Chicago problem 
//                                   fastest GPU-HC solver repository.
//
// ===================================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <vector>

// magma
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_internal.h"
/*#undef max
#undef min
#include "magma_templates.h"
#include "sync.cuh"
#undef max
#undef min
#include "shuffle.cuh"
#undef max
#undef min
#include "batched_kernel_param.h"*/

#include "Problem_Params.hpp"
#include "./gpu_solver/gpu-kernels/magmaHC-kernels.h"

namespace magmaHCWrapper {

  void HC_Track_Chicago_Problem(
    magmaFloatComplex *h_startSols, magmaFloatComplex *h_Track,
    magmaFloatComplex *h_startParams, magmaFloatComplex *h_targetParams,
    magma_int_t *h_Hx_idx, magma_int_t *h_Ht_idx,
    Problem_Params* pp, std::string hc_problem, 
    std::ofstream &GPUHC_Solution_File)
  {
    magma_init();
    magma_print_environment();

    magma_int_t batchCount = pp->numOfTracks;
    magma_int_t N = pp->numOfVars;

    real_Double_t     gpu_time;
    magmaFloatComplex *h_cgesvA, *h_cgesvB;
    magmaFloatComplex *h_cgesvA_verify, *h_cgesvB_verify;
    magmaFloatComplex *h_track_sols;
    magmaFloatComplex_ptr d_startSols, d_Track;
    magmaFloatComplex *d_startParams, *d_targetParams;
    magmaFloatComplex_ptr d_cgesvA, d_cgesvB;
    magma_int_t ldda, lddb, ldd_params, sizeA, sizeB;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magmaFloatComplex **d_startSols_array = NULL;
    magmaFloatComplex **d_Track_array = NULL;
    magmaFloatComplex **d_cgesvA_array = NULL;
    magmaFloatComplex **d_cgesvB_array = NULL;

    magma_device_t cdev;       // variable to indicate current gpu id
    magma_queue_t my_queue;    // magma queue variable, internally holds a cuda stream and a cublas handle
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &my_queue );     // create a queue on this cdev

    ldda   = magma_roundup( N, 32 );  // multiple of 32 by default
    lddb   = ldda;
    ldd_params = pp->numOfParams + 1;     //> padded with a dummy parameter
    sizeA = N*N*batchCount;
    sizeB = N*batchCount;

    // ==================================================================================================
    // -- Hx and Ht index matrices --
    // ==================================================================================================
    magma_int_t *d_Hx_idx;
    magma_int_t *d_Ht_idx;
    magma_int_t size_Hx = N*N*pp->Hx_maximal_terms*(pp->Hx_maximal_parts);
    magma_int_t size_Ht = N*pp->Ht_maximal_terms*(pp->Ht_maximal_parts);
    magma_int_t rnded_size_Hx = magma_roundup( size_Hx, 32 );
    magma_int_t rnded_size_Ht = magma_roundup( size_Ht, 32 );

    // -- allocate gpu memories --
    magma_imalloc( &d_Hx_idx, size_Hx );
    magma_imalloc( &d_Ht_idx, size_Ht );

    // -- transfer data from cpu to gpu --
    magma_isetmatrix( size_Hx, 1, h_Hx_idx, size_Hx, d_Hx_idx, rnded_size_Hx, my_queue );
    magma_isetmatrix( size_Ht, 1, h_Ht_idx, size_Ht, d_Ht_idx, rnded_size_Ht, my_queue );
    // ==================================================================================================

    //> compute the parameter difference (target parameter - start parameter)
    //> On the CPU side
    magmaFloatComplex *h_params_diff;
    magma_cmalloc_cpu( &h_params_diff,  pp->numOfParams+1 );
    for(int i = 0; i < pp->numOfParams; i++) {
      (h_params_diff)[i] = (h_targetParams)[i] - (h_startParams)[i];
    }
    h_params_diff[pp->numOfParams] = MAGMA_C_ZERO;

    //> On the GPU side
    magmaFloatComplex *d_diffParams;
    magma_cmalloc( &d_diffParams, ldd_params );
    magma_csetmatrix( pp->numOfParams+1, 1, h_params_diff,  pp->numOfParams+1, d_diffParams, ldd_params, my_queue );
    
    // -- allocate CPU memory --
    magma_cmalloc_cpu( &h_cgesvA, N*N*batchCount );
    magma_cmalloc_cpu( &h_cgesvB, N*batchCount );
    magma_cmalloc_cpu( &h_cgesvA_verify, N*N*batchCount );
    magma_cmalloc_cpu( &h_cgesvB_verify, N*batchCount );
    magma_cmalloc_cpu( &h_track_sols, (N+1)*batchCount );

    int s = 0;

    // -- allocate GPU gm --
    magma_cmalloc( &d_startSols, (N+1)*batchCount );
    magma_cmalloc( &d_Track, (N+1)*batchCount );
    magma_cmalloc( &d_startParams, ldd_params );
    magma_cmalloc( &d_targetParams, ldd_params );
    magma_cmalloc( &d_cgesvA, ldda*N*batchCount );
    magma_cmalloc( &d_cgesvB, ldda*batchCount );

    // -- allocate 2d arrays in GPU gm --
    magma_malloc( (void**) &d_startSols_array,  batchCount * sizeof(magmaFloatComplex*) );
    magma_malloc( (void**) &d_Track_array,    batchCount * sizeof(magmaFloatComplex*) );
    magma_malloc( (void**) &d_cgesvA_array,    batchCount * sizeof(magmaFloatComplex*) );
    magma_malloc( (void**) &d_cgesvB_array,    batchCount * sizeof(magmaFloatComplex*) );

    // -- random initialization for h_cgesvA and h_cgesvB (doesn't matter the value) --
    lapackf77_clarnv( &ione, ISEED, &sizeA, h_cgesvA );
    lapackf77_clarnv( &ione, ISEED, &sizeB, h_cgesvB );

    // -- transfer data from CPU memory to GPU memory --
    magma_csetmatrix( N+1, batchCount, h_startSols, (N+1), d_startSols, (N+1), my_queue );
    magma_csetmatrix( N+1, batchCount, h_Track, (N+1), d_Track, (N+1), my_queue );
    magma_csetmatrix( pp->numOfParams+1, 1, h_startParams,  pp->numOfParams+1, d_startParams,  ldd_params, my_queue );
    magma_csetmatrix( pp->numOfParams+1, 1, h_targetParams, pp->numOfParams+1, d_targetParams, ldd_params, my_queue );
    magma_csetmatrix( N, N*batchCount, h_cgesvA, N, d_cgesvA, ldda, my_queue );
    magma_csetmatrix( N, batchCount,   h_cgesvB, N, d_cgesvB, lddb, my_queue );

    // -- connect pointer to 2d arrays --
    magma_cset_pointer( d_startSols_array, d_startSols, (N+1), 0, 0, (N+1), batchCount, my_queue );
    magma_cset_pointer( d_Track_array, d_Track, (N+1), 0, 0, (N+1), batchCount, my_queue );
    magma_cset_pointer( d_cgesvA_array, d_cgesvA, ldda, 0, 0, ldda*N, batchCount, my_queue );
    magma_cset_pointer( d_cgesvB_array, d_cgesvB, lddb, 0, 0, ldda, batchCount, my_queue );

    // ===================================================================
    // GPU-HC
    // ===================================================================
    std::cout<<"Solving trifocal relative pose estimation with 2 oriented points and 1 point correspondences 30x30 problem"<<std::endl;
    std::cout<<"using direct parameter homotopy evaluation ..."<<std::endl<<std::endl;
    
    /*gpu_time = kernel_HC_Solver_trifocal_2op1p_30_direct_param_homotopy
              (my_queue, ldda, N, pp->numOfParams, batchCount, d_startSols_array, d_Track_array, 
                d_startParams, d_targetParams, d_cgesvA_array, d_cgesvB_array,
                d_diffParams, d_Hx_idx, d_Ht_idx);*/
    

    // -- check returns from the kernel --
    magma_cgetmatrix( (N+1), batchCount, d_Track, (N+1), h_track_sols, (N+1), my_queue );
    magma_cgetmatrix( N, batchCount, d_cgesvB, lddb, h_cgesvB_verify, N, my_queue );
    magma_cgetmatrix( N, N*batchCount, d_cgesvA, ldda, h_cgesvA_verify, N, my_queue );

    int num_of_convergence = 0;
    for (int bs = 0; bs < batchCount; bs++) {
      GPUHC_Solution_File << std::setprecision(10);
      
      if (MAGMA_C_REAL((h_cgesvB_verify + bs * N)[0]) == 1) {

        //GPUHC_Solution_File << bs << "\t" << MAGMA_C_REAL((h_cgesvB_verify + bs*N)[0]) << "\t" << MAGMA_C_IMAG((h_cgesvB_verify + bs*N)[0]) << "\n";
        GPUHC_Solution_File << MAGMA_C_IMAG((h_cgesvB_verify + bs*N)[0]) << "\t" << std::setprecision(20) << (gpu_time)*1000 << "\n";
        for (int vs = 0; vs < N; vs++) {
          GPUHC_Solution_File << std::setprecision(20) << MAGMA_C_REAL((h_track_sols + bs * (N+1))[vs]) << "\t" << std::setprecision(20) << MAGMA_C_IMAG((h_track_sols + bs * (N+1))[vs]) << "\n";
        }
        GPUHC_Solution_File << "\n";
        num_of_convergence++;
      }
    }
    std::cout<< "Number of convergence: " << num_of_convergence <<std::endl;

    //> Show how much time GPU takes for solving the problem
    printf("============== GPU time (ms) ==============\n");
    printf("%7.2f (ms)\n", (gpu_time)*1000);
    printf("===========================================\n\n");

    magma_queue_destroy( my_queue );

    magma_free_cpu( h_cgesvA );
    magma_free_cpu( h_cgesvB );
    magma_free_cpu( h_cgesvA_verify );
    magma_free_cpu( h_cgesvB_verify );
    magma_free_cpu( h_track_sols );

    magma_free_cpu( h_params_diff );

    magma_free( d_startSols );
    magma_free( d_Track );
    magma_free( d_startParams );
    magma_free( d_targetParams );
    magma_free( d_cgesvA );
    magma_free( d_cgesvB );
    magma_free( d_startSols_array );
    magma_free( d_Track_array );
    magma_free( d_cgesvA_array );
    magma_free( d_cgesvB_array );
    magma_free( d_Hx_idx );
    magma_free( d_Ht_idx );

    magma_free( d_diffParams );

    fflush( stdout );
    printf( "\n" );
    magma_finalize();
  }

} // end of namespace

#endif
