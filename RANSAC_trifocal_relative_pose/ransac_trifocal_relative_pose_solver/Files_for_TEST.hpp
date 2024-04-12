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
#include "Views.hpp"

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

namespace TEST_WITH_WRITTEN_FILES {
    
    class write_files_for_test {

    public:
        write_files_for_test( );

        void write_GroundTruths( Eigen::Matrix3d R21, Eigen::Matrix3d R31, Eigen::Vector3d T21, Eigen::Vector3d T31 );
        
        void write_true_solution_HC_steps( std::vector<int> true_solution_hc_steps );

        void write_Pose_Residuals( Eigen::Vector3d R21_Residuals, Eigen::Vector3d R31_Residuals,
                                   Eigen::Vector3d T21_Residuals, Eigen::Vector3d T31_Residuals );

        void write_So_far_the_best_Pose( Eigen::Vector3d best_raw_r21, Eigen::Vector3d best_raw_r31,
                                         Eigen::Vector3d best_raw_t21, Eigen::Vector3d best_raw_t31 );

        void write_time_when_depths_are_positive( std::vector<float> time_cue );

        void write_block_cycle_times( std::vector<float> block_cycle_times );

        void write_final_information( std::vector<int> final_indices, TrifocalViewsWrapper::Trifocal_Views views,
                                      std::array<int, 3> final_match_indices, std::array<float, 18> final_depths,
                                      Eigen::Vector3d final_Unnormalized_R21, Eigen::Vector3d final_Unnormalized_R31, 
                                      Eigen::Vector3d final_Unnormalized_T21, Eigen::Vector3d final_Unnormalized_T31);

        void close_all_files();

    private:

        std::ofstream GroundTruth_Pose_File;

        std::ofstream SoFarTheBest_Pose_R21_File;
        std::ofstream SoFarTheBest_Pose_R31_File;
        std::ofstream SoFarTheBest_Pose_T21_File;
        std::ofstream SoFarTheBest_Pose_T31_File;

        std::ofstream Pose_Residuals_R21_File;
        std::ofstream Pose_Residuals_R31_File;
        std::ofstream Pose_Residuals_T21_File;
        std::ofstream Pose_Residuals_T31_File;

        std::ofstream Final_Result_Information;
        
        std::ofstream HC_Steps_File;
        std::ofstream Positive_Depths_File;
        std::ofstream Block_Cycle_Times_File;
    };

}


#endif
