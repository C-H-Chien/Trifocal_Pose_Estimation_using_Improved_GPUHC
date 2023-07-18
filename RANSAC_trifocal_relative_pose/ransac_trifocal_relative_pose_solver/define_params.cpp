#ifndef define_params_and_read_files_cpp
#define define_params_and_read_files_cpp
// ==============================================================================
//
// Modifications
//    Chiang-Heng Chien  22-10-31:   Initially Created (Copied from other repos)
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ==============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

#include "Problem_Params.hpp"

namespace magmaHCWrapper {

  void Problem_Params::define_problem_params(std::string problem_filename, std::string HC_problem)
  {
    if (HC_problem == "trifocal_2op1p") {

      //> problem specifications
      /*numOfParams = 33;
      numOfTracks = 312;
      numOfVars = 30;
      numOfCoeffsFromParams = 37;

      //> Indexing evaluations
      Hx_maximal_terms = 8;
      Hx_maximal_parts = 4;
      Ht_maximal_terms = 16;
      Ht_maximal_parts = 5;

      max_orderOf_t = 2;*/

      //> problem specifications
      numOfParams = 33;
      numOfTracks = 312;
      numOfVars = 30;
      
      //> constant matrices parameters --
      Hx_maximal_terms = 8;
      Hx_maximal_parts = 5;
      Ht_maximal_terms = 16;
      Ht_maximal_parts = 6;

      numOfCoeffsFromParams = 92; //> don't care 
      max_orderOf_t = 2;
    }
    else {
      std::cout<<"You are entering invalid HC problem in your input argument!"<<std::endl;
      exit(1);
    }
  }
} // end of namespace

#endif
