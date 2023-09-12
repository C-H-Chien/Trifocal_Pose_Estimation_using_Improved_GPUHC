#ifndef FILES_FOR_TEST_HPP
#define FILES_FOR_TEST_HPP
// =============================================================================
//
// Modifications
//    Chiang-Heng Chien  23-08-01:   Intiailly Created for Multiview Geometry
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ==============================================================================
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

#include "definitions.h"

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

namespace TEST_WITH_WRITTEN_FILES {
    
    class write_files_for_test {

    public:
        write_files_for_test( );
        
        void write_true_solution_HC_steps( std::vector<int> true_solution_hc_steps );

        void close_all_files();

    private:

        std::ofstream HC_Steps_File;
        
    };

}


#endif
