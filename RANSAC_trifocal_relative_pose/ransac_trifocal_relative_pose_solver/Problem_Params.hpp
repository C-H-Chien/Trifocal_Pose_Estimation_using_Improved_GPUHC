#ifndef magmaHC_problems_cuh
#define magmaHC_problems_cuh
// =======================================================================
//
// Modifications
//    Chiang-Heng Chien  21-10-12:   Intiailly Created
//
// Notes
//    Chiang-Heng Chien  22-11-12:   (TODO) This script has to be reorganized.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "magma_v2.h"

namespace magmaHCWrapper {

    // -- informaiton of the benchmark problem --
    class Problem_Params {
    public:
      int numOfTracks;
      int numOfParams;
      int numOfVars;
      int numOfCoeffsFromParams;

      int Hx_maximal_terms;
      int Hx_maximal_parts;
      int Ht_maximal_terms;
      int Ht_maximal_parts;

      int max_orderOf_t;

      void define_problem_params(std::string problem_filename, std::string HC_problem);
    };

    void print_usage();

    void HC_Track_General_Problem(
      magmaFloatComplex *h_startSols, magmaFloatComplex *h_Track,
      magmaFloatComplex *h_startParams, magmaFloatComplex *h_targetParams,
      magma_int_t *h_Hx_idx, magma_int_t *h_Ht_idx,
      magmaFloatComplex *h_phc_coeffs_H, magmaFloatComplex *h_phc_coeffs_Ht,
      Problem_Params* pp, std::string hc_problem, 
      std::ofstream &GPUHC_Solution_File
    );

    void HC_Track_Chicago_Problem(
      magmaFloatComplex *h_startSols, magmaFloatComplex *h_Track,
      magmaFloatComplex *h_startParams, magmaFloatComplex *h_targetParams,
      magma_int_t *h_Hx_idx, magma_int_t *h_Ht_idx,
      Problem_Params* pp, std::string hc_problem, 
      std::ofstream &GPUHC_Solution_File
    );
}

#endif
