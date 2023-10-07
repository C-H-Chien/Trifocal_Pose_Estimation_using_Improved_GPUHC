#ifndef RANSAC_SYSTEM_MB_HPP
#define RANSAC_SYSTEM_MB_HPP
// =============================================================================================
//
// Modifications
//    Chiang-Heng Chien  23-07-05:   Intiailly Created. Wrap up GPU-HC solver under a RANSAC scheme.
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

#include <Eigen/Core>
#include <Eigen/Dense>

#include "magma_v2.h"
#include "Problem_Params.hpp"
#include "Views.hpp"

//> TESTING
//typedef Eigen::Matrix<float, 8, 2> vec_Depths;


namespace RANSAC_Estimator {
    
    class RANSAC_System_MB {

    public:
        
        RANSAC_System_MB( magmaHCWrapper::Problem_Params* );

        void Array_Memory_Allocations( magmaHCWrapper::Problem_Params* pp, bool is_param_pad_with_ZERO );
        void Prepare_Target_Params( TrifocalViewsWrapper::Trifocal_Views views, magmaHCWrapper::Problem_Params* pp );
        void Transfer_Data_From_CPU_to_GPU();
        void Solve_Relative_Pose( magmaHCWrapper::Problem_Params* pp );
        void Transfer_Data_From_GPU_to_CPU();
        void Write_Solutions_To_Files( std::ofstream &GPUHC_Solution_File );
        void Transform_Solutions_To_Relative_Poses( TrifocalViewsWrapper::Trifocal_Views views );
        void Find_Solution_With_Maximal_Inliers( TrifocalViewsWrapper::Trifocal_Views views );
        bool Solution_Residual_From_GroundTruths( TrifocalViewsWrapper::Trifocal_Views views );
        void Push_Block_Cycle_Time();
        void Free_Memories();

        //void RANSAC_Relative_Pose_by_

        magmaFloatComplex *h_startSols;
        magmaFloatComplex *h_Track;
        magmaFloatComplex *h_startParams;
        magmaFloatComplex *h_targetParams;
        magmaFloatComplex *h_phc_coeffs_Hx;
        magmaFloatComplex *h_phc_coeffs_Ht;
        magma_int_t       *h_Hx_idx;
        magma_int_t       *h_Ht_idx;

        Eigen::Matrix3d final_R21;
        Eigen::Matrix3d final_R31;
        Eigen::Vector3d final_T21;
        Eigen::Vector3d final_T31;

        //> Rotation and Translation Residuals
        Eigen::Vector3d R21_Residual;
        Eigen::Vector3d R31_Residual;
        Eigen::Vector3d T21_Residual;
        Eigen::Vector3d T31_Residual;

        //> Stack all residuals
        std::vector< Eigen::Vector3d > Stacked_R21_Residuals;
        std::vector< Eigen::Vector3d > Stacked_R31_Residuals;
        std::vector< Eigen::Vector3d > Stacked_T21_Residuals;
        std::vector< Eigen::Vector3d > Stacked_T31_Residuals;

        //> Timings
        real_Double_t     gpu_time;

        //> TEST OF BLOCK CYCLE TIMINGS
        long long *clk_data;
        long long *host_clk_data;
        std::vector< float > all_Block_Cycle_Times;

        //> Some TESTINGS
        std::vector< int > true_solution_hc_steps;
        std::vector< int > real_solution_hc_steps;
        std::vector< float > t_when_depths_are_positive; 
        std::vector< float > t_when_depths_are_positive_for_actual_sol; 
        
    private:
        int Number_Of_Points;
        std::string problem_FileName;

        magma_device_t cdev;       // variable to indicate current gpu id
        magma_queue_t my_queue;    // magma queue variable, internally holds a cuda stream and a cublas handle
        int peak_clk;
        cudaError_t err;

        magma_int_t batchCount;
        magma_int_t N;

        magmaFloatComplex *h_cgesvA, *h_cgesvB;
        magmaFloatComplex *h_cgesvA_verify, *h_cgesvB_verify;
        magmaFloatComplex *h_path_converge_flag;
        magmaFloatComplex *h_track_sols;
        magmaFloatComplex_ptr d_startSols, d_Track;
        magmaFloatComplex *d_startParams, *d_targetParams;
        magmaFloatComplex_ptr d_cgesvA, d_cgesvB;
        magmaFloatComplex *d_path_converge_flag;
        magma_int_t ldda, lddb, lddc, ldd_params, sizeA, sizeB;
        magma_int_t ione;

        magmaFloatComplex **d_startSols_array;
        magmaFloatComplex **d_Track_array;
        magmaFloatComplex **d_cgesvA_array;
        magmaFloatComplex **d_cgesvB_array;

        magma_int_t *d_Hx_idx;
        magma_int_t *d_Ht_idx;
        magma_int_t size_Hx;
        magma_int_t size_Ht;
        magma_int_t rnded_size_Hx;
        magma_int_t rnded_size_Ht;

        magmaFloatComplex *h_params_diff;
        magmaFloatComplex *d_diffParams;

        Eigen::Vector3d r21;
        Eigen::Vector3d r31;
        std::vector< Eigen::Vector3d > normalized_t21;
        std::vector< Eigen::Vector3d > normalized_t31;
        std::vector< Eigen::Matrix3d > R21;
        std::vector< Eigen::Matrix3d > R31;
        std::vector< Eigen::Matrix3d > E21;
        std::vector< Eigen::Matrix3d > E31;
        std::vector< Eigen::Matrix3d > F21;
        std::vector< Eigen::Matrix3d > F31;
        Eigen::Vector3d t21;
        Eigen::Vector3d t31;

        //> TESTING
        std::vector< std::vector<float> > Depths_real;
        std::vector< std::vector<float> > Depths_imag;
    };

}


#endif
