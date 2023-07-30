#ifndef HC_DATA_READER_CPP
#define HC_DATA_READER_CPP
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

#include "HC_data_reader.hpp"
#include "magma_v2.h"
#include "definitions.h"
#include "Problem_Params.hpp"

namespace HC_Data_Import {
    
    HC_Data_Reader::HC_Data_Reader(
        std::string problem_FileName)
    :problem_FileName(problem_FileName) 
    {
        start_sols_read_success    = false;
        start_params_read_success  = false;
        target_params_read_success = false;
        Hx_file_read_success       = false;
        Ht_file_read_success       = false;
    }

    bool HC_Data_Reader::Read_Start_System_Solutions( magmaFloatComplex* &h_startSols, magmaFloatComplex* &h_Track, 
                                                      magmaHCWrapper::Problem_Params* pp, bool use_MB ) 
    {
        int loop_end_for_startSols = ( use_MB ) ? MULTIPLES_OF_BATCHCOUNT : ( 1 );
        //int loop_end_for_Track     = ( use_MB ) ? : RANSAC_Number_Of_Iterations;

        int d = 0, i = 0; 
        float s_real, s_imag;
        for (int mi = 0; mi < loop_end_for_startSols; mi++) {
            std::string startSols_File_Path   = problem_FileName + "/start_sols.txt";
            std::fstream startSols_file;
            d = 0, i = 0; 
            startSols_file.open(startSols_File_Path, std::ios_base::in);
            if (!startSols_file) { std::cerr << "problem start solutions file not existed!\n"; exit(1); }
            else {
                while (startSols_file >> s_real >> s_imag) {
                    (h_startSols + mi * (pp->numOfTracks * (pp->numOfVars+1)) + i * (pp->numOfVars+1))[d] = MAGMA_C_MAKE(s_real, s_imag);
                    if (d < pp->numOfVars-1) { d++; }
                    else {
                        d = 0;
                        i++;
                    }
                }
                for(int k = 0; k < pp->numOfTracks; k++) {
                    (h_startSols + mi * (pp->numOfTracks * (pp->numOfVars+1)) + k * (pp->numOfVars+1))[pp->numOfVars] = MAGMA_C_MAKE(1.0, 0.0);
                }
                assert( i == pp->numOfTracks * pp->numOfVars );
                start_sols_read_success = true;
            }
        }
        

        for (int ri = 0; ri < RANSAC_Number_Of_Iterations; ri++) {
            d = 0, i = 0; 
            std::string startSols_File_Path_   = problem_FileName + "/start_sols.txt";
            std::fstream startSols_file_;
            startSols_file_.open(startSols_File_Path_, std::ios_base::in);
            if (!startSols_file_) { std::cerr << "Path " << startSols_File_Path_ << " Not Exist!\n"; exit(1); }
            else {
                while (startSols_file_ >> s_real >> s_imag) {
                    (h_Track     + ri * (pp->numOfTracks * (pp->numOfVars+1)) + i * (pp->numOfVars+1))[d] = MAGMA_C_MAKE(s_real, s_imag);
                    if (d < pp->numOfVars-1) { d++; }
                    else {
                        d = 0;
                        i++;
                    }
                }
                for(int k = 0; k < pp->numOfTracks; k++) {
                    (h_Track     + ri * (pp->numOfTracks * (pp->numOfVars+1)) + k * (pp->numOfVars+1))[pp->numOfVars] = MAGMA_C_MAKE(1.0, 0.0);
                }
                assert( i == pp->numOfTracks * pp->numOfVars );
                start_sols_read_success = true;
            }
        }

        //std::cout << "Start solution from HC data reader:" << std::endl;
        //magma_cprint((pp->numOfVars+1), 1, (h_Track + (pp->numOfTracks * (pp->numOfVars+1)) + 0*(pp->numOfVars+1) ), (pp->numOfVars+1));

        return start_sols_read_success;
    }

    //> 
    bool HC_Data_Reader::Read_Start_System_Parameters(magmaFloatComplex* &h_startParams, magmaHCWrapper::Problem_Params* pp, bool need_to_pad_ONE) 
    {
        std::string startParams_File_Path = problem_FileName + "/start_params.txt";
        std::fstream startParams_File;
        int d = 0;
        float s_real, s_imag;
        startParams_File.open(startParams_File_Path, std::ios_base::in);
        if (!startParams_File) { std::cerr << "problem start parameters file not existed!\n"; exit(1); }
        else {
            while (startParams_File >> s_real >> s_imag) {
                (h_startParams)[d] = MAGMA_C_MAKE(s_real, s_imag);
                d++;
            }
            if (need_to_pad_ONE) (h_startParams)[d] = MAGMA_C_MAKE(1.0, 0.0);
            assert( d == pp->numOfParams+1 );
            start_params_read_success = true;
        }

        return start_params_read_success;
    }

    bool HC_Data_Reader::Read_Hx_Ht_Indices(magma_int_t* &h_Hx_idx, magma_int_t* &h_Ht_idx, magmaHCWrapper::Problem_Params* pp)
    {
        std::string filename_Hx = problem_FileName;
        std::string filename_Ht = problem_FileName;
        filename_Hx.append("/Hx_indices.txt");
        filename_Ht.append("/Ht_indices.txt");
        std::fstream Hx_idx_file;
        std::fstream Ht_idx_file;

        //> Read Hx index matrix, if required
        int index;
        int d = 0;
        float s_real, s_imag;
        Hx_idx_file.open(filename_Hx, std::ios_base::in);
        if (!Hx_idx_file) { std::cout << "File path " << filename_Hx << " not existed!\n"; exit(1); }
        else {
            while (Hx_idx_file >> index) {
                (h_Hx_idx)[d] = index;
                d++;
            }
            Hx_file_read_success = true;
        }

        //> Read Ht index matrix
        d = 0;
        Ht_idx_file.open(filename_Ht, std::ios_base::in);
        if (!Ht_idx_file) { std::cout << "File path " << filename_Ht << " not existed!\n"; exit(1); }
        else {
            while (Ht_idx_file >> index) {
                (h_Ht_idx)[d] = index;
                d++;
            }
            Ht_file_read_success = true;
        }

        return Hx_file_read_success & Ht_file_read_success;
    }

    //> 
    bool HC_Data_Reader::Read_Target_System_Parameters(magmaFloatComplex* &h_targetParams, magmaHCWrapper::Problem_Params* pp, bool need_to_pad_ONE) 
    {
        std::string targetParams_File_Path = problem_FileName + "/target_params.txt";
        std::fstream targetParams_File;
        int d = 0;
        float s_real, s_imag;
        targetParams_File.open(targetParams_File_Path, std::ios_base::in);
        if (!targetParams_File) { std::cerr << "File Path " << targetParams_File_Path << "Does NOT Exist!\n"; exit(1); }
        else {
            while (targetParams_File >> s_real >> s_imag) {
                (h_targetParams)[d] = MAGMA_C_MAKE(s_real, s_imag);
                d++;
            }
            if (need_to_pad_ONE) (h_targetParams)[d] = MAGMA_C_MAKE(1.0, 0.0);
            assert( d == pp->numOfParams+1 );
            target_params_read_success = true;
        }

        return target_params_read_success;
    }

}

#endif
