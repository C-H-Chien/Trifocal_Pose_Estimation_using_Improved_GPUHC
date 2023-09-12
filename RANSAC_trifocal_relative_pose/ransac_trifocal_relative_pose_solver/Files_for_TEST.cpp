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
        std::string write_file_dir = REPO_DIR + "HC_steps.txt";
        HC_Steps_File.open(write_file_dir);
        if ( !HC_Steps_File.is_open() ) std::cout << "files " << write_file_dir << " cannot be opened!" << std::endl;

    }

    void write_files_for_test::write_true_solution_HC_steps( std::vector<int> true_solution_hc_steps )
    {
        for (int i = 0; i < true_solution_hc_steps.size(); i++ ) {
            HC_Steps_File << true_solution_hc_steps[i] << "\n";
        }
    }

    void write_files_for_test::close_all_files() {
        HC_Steps_File.close();
    }
}

#endif
