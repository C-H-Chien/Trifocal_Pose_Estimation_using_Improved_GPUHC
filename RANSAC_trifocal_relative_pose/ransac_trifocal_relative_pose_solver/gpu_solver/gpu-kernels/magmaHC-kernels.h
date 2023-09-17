#ifndef magmaHC_kernels_h
#define magmaHC_kernels_h
// ============================================================================
// Header file declaring all kernels
//
// Modifications
//    Chiang-Heng Chien  22-10-31:   Initially Created (Copied from other repos)
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

// -- magma --
#include "flops.h"
#include "magma_v2.h"

extern "C" {
namespace magmaHCWrapper {

  /*real_Double_t 
  kernel_HC_Solver_trifocal_2op1p_30_direct_param_homotopy(
    magma_queue_t my_queue,
    magma_int_t ldda,
    magma_int_t N, 
    magma_int_t num_of_params,
    magma_int_t batchCount, 
    magmaFloatComplex** d_startSols_array, 
    magmaFloatComplex** d_Track_array,
    magmaFloatComplex*  d_startParams,
    magmaFloatComplex*  d_targetParams,
    magmaFloatComplex** d_cgesvA_array, 
    magmaFloatComplex** d_cgesvB_array,
    magmaFloatComplex*  d_diffParams,
    magma_int_t* d_Hx_indx, 
    magma_int_t* d_Ht_indx,
    magmaFloatComplex*  d_path_converge_flag
  );*/

  real_Double_t 
  kernel_HC_Solver_trifocal_2op1p_30_direct_param_homotopy_mb(
    magma_queue_t my_queue,
    magma_int_t ldda,
    magma_int_t N, 
    magma_int_t num_of_params,
    magma_int_t batchCount, 
    magmaFloatComplex** d_startSols_array, 
    magmaFloatComplex** d_Track_array,
    magmaFloatComplex*  d_startParams,
    magmaFloatComplex*  d_targetParams,
    magmaFloatComplex** d_cgesvA_array, 
    magmaFloatComplex** d_cgesvB_array,
    magmaFloatComplex*  d_diffParams,
    magma_int_t* d_Hx_indx, 
    magma_int_t* d_Ht_indx,
    magmaFloatComplex*  d_path_converge_flag
  );

  //> Kernel that measures block cycle time
  real_Double_t 
  kernel_HC_Solver_trifocal_2op1p_30_direct_param_homotopy_cycle_timing(
    magma_queue_t my_queue,
    magma_int_t ldda,
    magma_int_t N, 
    magma_int_t num_of_params,
    magma_int_t batchCount, 
    magmaFloatComplex** d_startSols_array, 
    magmaFloatComplex** d_Track_array,
    magmaFloatComplex*  d_startParams,
    magmaFloatComplex*  d_targetParams,
    magmaFloatComplex** d_cgesvA_array, 
    magmaFloatComplex** d_cgesvB_array,
    magmaFloatComplex*  d_diffParams,
    magma_int_t* d_Hx_indx, 
    magma_int_t* d_Ht_indx,
    magmaFloatComplex*  d_path_converge_flag,
    long long *clocks
  );
}
}

#endif
