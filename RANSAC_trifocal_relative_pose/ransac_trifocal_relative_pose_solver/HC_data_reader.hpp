#ifndef HC_DATA_READER_HPP
#define HC_DATA_READER_HPP
// =============================================================================================
//
// Modifications
//    Chiang-Heng Chien  23-07-04:   Intiailly Created. Shift reading files for the GPU-HC solver 
//                                   from its original repository to a class structure here.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =============================================================================================
#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <vector>
#include <chrono>

#include "magma_v2.h"
#include "Problem_Params.hpp"

namespace HC_Data_Import {
    
    class HC_Data_Reader {

    public:
        
        HC_Data_Reader(std::string);
        bool Read_Start_System_Solutions(magmaFloatComplex* &h_startSols, magmaFloatComplex* &h_Track, magmaHCWrapper::Problem_Params* pp);
        bool Read_Start_System_Parameters(magmaFloatComplex* &h_startParams, magmaHCWrapper::Problem_Params* pp, bool need_to_pad_ONE);
        bool Read_Hx_Ht_Indices(magma_int_t* &h_Hx_idx, magma_int_t* &h_Ht_idx, magmaHCWrapper::Problem_Params* pp);

        //> Used for Debugging
        bool Read_Target_System_Parameters(magmaFloatComplex* &h_targetParams, magmaHCWrapper::Problem_Params* pp, bool need_to_pad_ONE);
        
    private:
        std::string problem_FileName;
        bool start_sols_read_success;
        bool start_params_read_success;
        bool target_params_read_success;
        bool Hx_file_read_success;
        bool Ht_file_read_success;
    };

}


#endif
