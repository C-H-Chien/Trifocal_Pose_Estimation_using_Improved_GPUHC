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
        //> Ground Truth Pose file
        std::string write_file_dir_GT_Pose = REPO_DIR + "GroundTruth_Pose.txt";
        GroundTruth_Pose_File.open(write_file_dir_GT_Pose);
        if ( !GroundTruth_Pose_File.is_open() ) std::cout << "File " << write_file_dir_GT_Pose << " cannot be opened!" << std::endl;

        std::string write_file_dir_Result_Information = REPO_DIR + "Result_Information.txt";
        Final_Result_Information.open(write_file_dir_Result_Information);
        if ( !Final_Result_Information.is_open() ) std::cout << "File " << write_file_dir_Result_Information << " cannot be opened!" << std::endl;

        //> Pose Residuals File
        std::string write_file_dir_Pose_Residuals_R21 = REPO_DIR + "Stacked_Residuals_R21.txt";
        Pose_Residuals_R21_File.open(write_file_dir_Pose_Residuals_R21);
        if ( !Pose_Residuals_R21_File.is_open() ) std::cout << "File " << write_file_dir_Pose_Residuals_R21 << " cannot be opened!" << std::endl;

        std::string write_file_dir_Pose_Residuals_R31 = REPO_DIR + "Stacked_Residuals_R31.txt";
        Pose_Residuals_R31_File.open(write_file_dir_Pose_Residuals_R31);
        if ( !Pose_Residuals_R31_File.is_open() ) std::cout << "File " << write_file_dir_Pose_Residuals_R31 << " cannot be opened!" << std::endl;

        std::string write_file_dir_Pose_Residuals_T21 = REPO_DIR + "Stacked_Residuals_T21.txt";
        Pose_Residuals_T21_File.open(write_file_dir_Pose_Residuals_T21);
        if ( !Pose_Residuals_T21_File.is_open() ) std::cout << "File " << write_file_dir_Pose_Residuals_T21 << " cannot be opened!" << std::endl;

        std::string write_file_dir_Pose_Residuals_T31 = REPO_DIR + "Stacked_Residuals_T31.txt";
        Pose_Residuals_T31_File.open(write_file_dir_Pose_Residuals_T31);
        if ( !Pose_Residuals_T31_File.is_open() ) std::cout << "File " << write_file_dir_Pose_Residuals_T31 << " cannot be opened!" << std::endl;

        //> HC steps for true HC solution write file
        #if TEST_COLLECT_HC_STEPS
        std::string write_file_dir_HC_steps = REPO_DIR + "HC_steps.txt";
        HC_Steps_File.open(write_file_dir_HC_steps);
        if ( !HC_Steps_File.is_open() ) std::cout << "File " << write_file_dir_HC_steps << " cannot be opened!" << std::endl;
        #endif

        //> Test early stop mechanism from all positive depths
        #if TEST_ALL_POSITIVE_DEPTHS_AT_END_ZONE
        std::string write_file_dir_positive_depths = REPO_DIR + "bool_positive_depths.txt";
        Positive_Depths_File.open(write_file_dir_positive_depths);
        if ( !Positive_Depths_File.is_open() ) std::cout << "File " << write_file_dir_positive_depths << " cannot be opened!" << std::endl;
        #endif

        //> Write block cycle times
        #if TEST_BLOCK_CYCLE_TIME
        std::string write_file_dir_block_cycle_time = REPO_DIR + "Block_Cycle_Times.txt";
        Block_Cycle_Times_File.open(write_file_dir_block_cycle_time);
        if ( !Block_Cycle_Times_File.is_open() ) std::cout << "File " << write_file_dir_block_cycle_time << " cannot be opened!" << std::endl;
        #endif
    }

    void write_files_for_test::write_GroundTruths( Eigen::Matrix3d R21, Eigen::Matrix3d R31, Eigen::Vector3d T21, Eigen::Vector3d T31 )
    {
        //> Write R21
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                GroundTruth_Pose_File << R21(i,j) << "\t";
            }
            GroundTruth_Pose_File << "\n";
        }
        GroundTruth_Pose_File << "\n";

        //> Write R31
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                GroundTruth_Pose_File << R31(i,j) << "\t";
            }
            GroundTruth_Pose_File << "\n";
        }
        GroundTruth_Pose_File << "\n";

        //> Write T21
        for (int i = 0; i < 3; i++) GroundTruth_Pose_File << T21(i) << "\t";
        GroundTruth_Pose_File << "\n";

        //> Write T31
        for (int i = 0; i < 3; i++) GroundTruth_Pose_File << T31(i) << "\t";
        GroundTruth_Pose_File << "\n";
    }

    void write_files_for_test::write_true_solution_HC_steps( std::vector<int> true_solution_hc_steps )
    {
        //std::cout << true_solution_hc_steps.size() << std::endl;
        for (int i = 0; i < true_solution_hc_steps.size(); i++ ) {
            HC_Steps_File << true_solution_hc_steps[i] << "\n";
        }
    }

    void write_files_for_test::write_Pose_Residuals( 
         std::vector<Eigen::Vector3d> Stacked_R21_Residuals, std::vector<Eigen::Vector3d> Stacked_R31_Residuals,
         std::vector<Eigen::Vector3d> Stacked_T21_Residuals, std::vector<Eigen::Vector3d> Stacked_T31_Residuals )
    {
        for (int i = 0; i < Stacked_R21_Residuals.size(); i++) {
            for (int j = 0; j < 3; j++) {
                Pose_Residuals_R21_File << Stacked_R21_Residuals[i](j) << "\t";
                Pose_Residuals_R31_File << Stacked_R31_Residuals[i](j) << "\t";
                Pose_Residuals_T21_File << Stacked_T21_Residuals[i](j) << "\t"; 
                Pose_Residuals_T31_File << Stacked_T31_Residuals[i](j) << "\t";
            }
            Pose_Residuals_R21_File << "\n";
            Pose_Residuals_R31_File << "\n";
            Pose_Residuals_T21_File << "\n";
            Pose_Residuals_T31_File << "\n";
        }
    }

    void write_files_for_test::write_time_when_depths_are_positive( std::vector<float> time_cue ) {
        for (int i = 0; i < time_cue.size(); i++ ) Positive_Depths_File << time_cue[i] << "\n";
    }

    void write_files_for_test::write_block_cycle_times( std::vector<float> cycle_clock_times ) {
        for (int i = 0; i < cycle_clock_times.size(); i++ ) Block_Cycle_Times_File << cycle_clock_times[i] << "\n";
    }

    void write_files_for_test::write_final_information( std::vector<int> final_indices, 
                                                        Eigen::Matrix3d final_R21, Eigen::Matrix3d final_R31, 
                                                        Eigen::Vector3d final_T21, Eigen::Vector3d final_T31)
    {
        //> Write inlier indices
        for (int i = 0; i < final_indices.size(); i++) Final_Result_Information << final_indices[i] << "\t";
        Final_Result_Information << "\n";

        //> Write R21
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Final_Result_Information << final_R21(i,j) << "\t";
            }
            Final_Result_Information << "\n";
        }
        Final_Result_Information << "\n";

        //> Write R31
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Final_Result_Information << final_R31(i,j) << "\t";
            }
            Final_Result_Information << "\n";
        }
        Final_Result_Information << "\n";

        //> Write T21
        for (int i = 0; i < 3; i++) Final_Result_Information << final_T21[i] << "\t";
        Final_Result_Information << "\n";

        //> Write T31
        for (int i = 0; i < 3; i++) Final_Result_Information << final_T31[i] << "\t";
        Final_Result_Information << "\n\n";
    }

    void write_files_for_test::close_all_files() {
        GroundTruth_Pose_File.close();

        #if TEST_COLLECT_HC_STEPS
        HC_Steps_File.close();
        #endif

        #if TEST_ALL_POSITIVE_DEPTHS_AT_END_ZONE
        Positive_Depths_File.close();
        #endif

        #if TEST_BLOCK_CYCLE_TIME
        Block_Cycle_Times_File.close();
        #endif

        Pose_Residuals_R21_File.close();
        Pose_Residuals_R31_File.close();
        Pose_Residuals_T21_File.close();
        Pose_Residuals_T31_File.close();

        Final_Result_Information.close();
    }
}

#endif
