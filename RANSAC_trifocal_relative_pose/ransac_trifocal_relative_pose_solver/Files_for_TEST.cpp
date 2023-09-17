#ifndef FILES_FOR_TEST_CPP
#define FILES_FOR_TEST_CPP
// ====================================================================================================
//
// Modifications
//    Chiang-Heng Chien  23-08-01:   Intiailly Created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =====================================================================================================
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

#include "Files_for_TEST.hpp"

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

namespace TEST_WITH_WRITTEN_FILES {
    
    write_files_for_test::write_files_for_test( )
    {
        //> HC steps for true HC solution write file
        std::string write_file_dir_HC_steps = REPO_DIR + "HC_steps.txt";
        HC_Steps_File.open(write_file_dir_HC_steps);
        if ( !HC_Steps_File.is_open() ) std::cout << "File " << write_file_dir_HC_steps << " cannot be opened!" << std::endl;

        //> Test early stop mechanism from all positive depths
        std::string write_file_dir_positive_depths = REPO_DIR + "bool_positive_depths.txt";
        Positive_Depths_File.open(write_file_dir_positive_depths);
        if ( !Positive_Depths_File.is_open() ) std::cout << "File " << write_file_dir_positive_depths << " cannot be opened!" << std::endl;

        //> Write block cycle times
        std::string write_file_dir_block_cycle_time = REPO_DIR + "Block_Cycle_Times.txt";
        Block_Cycle_Times_File.open(write_file_dir_block_cycle_time);
        if ( !Block_Cycle_Times_File.is_open() ) std::cout << "File " << write_file_dir_block_cycle_time << " cannot be opened!" << std::endl;
    }

    void write_files_for_test::write_true_solution_HC_steps( std::vector<int> true_solution_hc_steps )
    {
        for (int i = 0; i < true_solution_hc_steps.size(); i++ ) {
            HC_Steps_File << true_solution_hc_steps[i] << "\n";
        }
    }

    void write_files_for_test::write_time_when_depths_are_positive( std::vector<float> time_cue )
    {
        for (int i = 0; i < time_cue.size(); i++ ) {
            Positive_Depths_File << time_cue[i] << "\n";
        }
    }

    void write_files_for_test::write_block_cycle_times( std::vector<float> cycle_clock_times ) {
        for (int i = 0; i < cycle_clock_times.size(); i++ ) {
            Block_Cycle_Times_File << cycle_clock_times[i] << "\n";
        }
    }

    void write_files_for_test::close_all_files() {
        HC_Steps_File.close();
        Positive_Depths_File.close();
    }
}

#endif
