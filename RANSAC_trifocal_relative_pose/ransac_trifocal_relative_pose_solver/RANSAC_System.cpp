#ifndef RANSAC_SYSTEM_CPP
#define RANSAC_SYSTEM_CPP
// =============================================================================================
//
// Modifications
//    Chiang-Heng Chien  23-07-05:   Intiailly Created. RANSAC_System class member functions. 
//                                   Wrap up GPU-HC solver under a RANSAC scheme.
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
#include <stdlib.h>

#include "RANSAC_System.hpp"
#include "magma_v2.h"
#include "definitions.h"
#include "Problem_Params.hpp"
#include "Views.hpp"
#include "util.hpp"
#include "./gpu_solver/gpu-kernels/magmaHC-kernels.h"

#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_internal.h"

namespace RANSAC_Estimator {
    
    RANSAC_System::RANSAC_System( magmaHCWrapper::Problem_Params* pp )
    {
        magma_init();
        magma_print_environment();

        batchCount = pp->numOfTracks;
        N = pp->numOfVars;
        ione = 1;

        d_startSols_array = NULL;
        d_Track_array = NULL;
        d_cgesvA_array = NULL;
        d_cgesvB_array = NULL;

        magma_getdevice( &cdev );
        magma_queue_create( cdev, &my_queue ); 

        ldda = magma_roundup( N, 32 );  // multiple of 32 by default
        lddb = ldda;
        
        sizeA   = N*N*batchCount;
        sizeB   = N*batchCount;
        size_Hx = N*N*pp->Hx_maximal_terms*(pp->Hx_maximal_parts);
        size_Ht = N*pp->Ht_maximal_terms*(pp->Ht_maximal_parts);
        rnded_size_Hx = magma_roundup( size_Hx, 32 );
        rnded_size_Ht = magma_roundup( size_Ht, 32 );
    }

    void RANSAC_System::Array_Memory_Allocations( magmaHCWrapper::Problem_Params* pp, bool is_param_pad_with_ZERO )
    {
        // ==================================================================================================
        // *CPU* Side
        // ==================================================================================================
        //> Necessary for all minimal solvers.
        magma_cmalloc_cpu( &h_startSols,     pp->numOfTracks*(pp->numOfVars+1) );
        magma_imalloc_cpu( &h_Hx_idx,        pp->numOfVars*pp->numOfVars*pp->Hx_maximal_terms*pp->Hx_maximal_parts );
        magma_imalloc_cpu( &h_Ht_idx,        pp->numOfVars*pp->Ht_maximal_terms*pp->Ht_maximal_parts );
        magma_cmalloc_cpu( &h_cgesvA,        N*N*batchCount );
        magma_cmalloc_cpu( &h_cgesvB,        N*batchCount );
        magma_cmalloc_cpu( &h_cgesvA_verify, N*N*batchCount );
        magma_cmalloc_cpu( &h_cgesvB_verify, N*batchCount );
        magma_cmalloc_cpu( &h_path_converge_flag, batchCount*RANSAC_Number_Of_Iterations );

        //> For a general solver, this is necessary, but for a trifocal relative pose estimation (Chicago), this is redundant.
        magma_cmalloc_cpu( &h_phc_coeffs_Hx, (pp->numOfCoeffsFromParams+1)*(pp->max_orderOf_t+1) );
        magma_cmalloc_cpu( &h_phc_coeffs_Ht, (pp->numOfCoeffsFromParams+1)*(pp->max_orderOf_t) );

        if ( is_param_pad_with_ZERO ) ldd_params = pp->numOfParams + 1;     //> padded with a dummy parameter
        else                          ldd_params = pp->numOfParams;

        magma_cmalloc_cpu( &h_Track,        (N+1)*batchCount*RANSAC_Number_Of_Iterations );
        magma_cmalloc_cpu( &h_track_sols,   (N+1)*batchCount*RANSAC_Number_Of_Iterations ); //> Used to store the return of the GPU-HC solutions
        magma_cmalloc_cpu( &h_startParams,  ldd_params );
        magma_cmalloc_cpu( &h_targetParams, ldd_params*RANSAC_Number_Of_Iterations );
        magma_cmalloc_cpu( &h_params_diff,  ldd_params*RANSAC_Number_Of_Iterations );

        // ==================================================================================================
        // *GPU* Side
        // ==================================================================================================
        magma_cmalloc( &d_startSols, (N+1)*batchCount );
        magma_cmalloc( &d_startParams, ldd_params );
        magma_cmalloc( &d_cgesvA, ldda*N*batchCount );
        magma_cmalloc( &d_cgesvB, ldda*batchCount );
        magma_imalloc( &d_Hx_idx, size_Hx );
        magma_imalloc( &d_Ht_idx, size_Ht );
        magma_cmalloc( &d_targetParams, ldd_params*RANSAC_Number_Of_Iterations );
        magma_cmalloc( &d_diffParams,   ldd_params*RANSAC_Number_Of_Iterations );
        magma_cmalloc( &d_Track,        (N+1)*batchCount*RANSAC_Number_Of_Iterations );

        magma_cmalloc( &d_path_converge_flag, batchCount*RANSAC_Number_Of_Iterations );

        //> Allocate 2d arrays in GPU Global Memory
        magma_malloc( (void**) &d_startSols_array, batchCount * sizeof(magmaFloatComplex*) );
        magma_malloc( (void**) &d_Track_array,     batchCount * RANSAC_Number_Of_Iterations * sizeof(magmaFloatComplex*) );
        magma_malloc( (void**) &d_cgesvA_array,    batchCount * sizeof(magmaFloatComplex*) );
        magma_malloc( (void**) &d_cgesvB_array,    batchCount * sizeof(magmaFloatComplex*) );

        //> Random initialization for h_cgesvA and h_cgesvB (doesn't matter the value)
        magma_int_t ISEED[4] = {0,0,0,1};
        lapackf77_clarnv( &ione, ISEED, &sizeA, h_cgesvA );
        lapackf77_clarnv( &ione, ISEED, &sizeB, h_cgesvB );        
    }

    void RANSAC_System::Prepare_Target_Params( TrifocalViewsWrapper::Trifocal_Views views, magmaHCWrapper::Problem_Params* pp )
    {
        int tidx;
        int p1_idx, p2_idx, p3_idx;
        srand (time(NULL));
        Number_Of_Points = views.cam1.img_points_meters.size();
        std::cout << Number_Of_Points << std::endl;

        std::cout << "Preparing target parameters ..." << std::endl;
        for (int ti = 0; ti < RANSAC_Number_Of_Iterations; ti++) {
            //> Pick 3 from All_Points_Indices without repeatition
            while(1) {
                p1_idx = rand() % Number_Of_Points;
                p2_idx = rand() % Number_Of_Points;
                p3_idx = rand() % Number_Of_Points;
                if ( (p1_idx != p2_idx) && (p1_idx != p3_idx) && (p2_idx != p3_idx) ) break;
            }

            (h_targetParams + ti*(pp->numOfParams+1))[0] = MAGMA_C_MAKE(views.cam1.img_points_meters[p1_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[1] = MAGMA_C_MAKE(views.cam1.img_points_meters[p1_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[2] = MAGMA_C_MAKE(views.cam2.img_points_meters[p1_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[3] = MAGMA_C_MAKE(views.cam2.img_points_meters[p1_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[4] = MAGMA_C_MAKE(views.cam3.img_points_meters[p1_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[5] = MAGMA_C_MAKE(views.cam3.img_points_meters[p1_idx](1), 0.0);

            (h_targetParams + ti*(pp->numOfParams+1))[6]  = MAGMA_C_MAKE(views.cam1.img_points_meters[p2_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[7]  = MAGMA_C_MAKE(views.cam1.img_points_meters[p2_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[8]  = MAGMA_C_MAKE(views.cam2.img_points_meters[p2_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[9]  = MAGMA_C_MAKE(views.cam2.img_points_meters[p2_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[10] = MAGMA_C_MAKE(views.cam3.img_points_meters[p2_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[11] = MAGMA_C_MAKE(views.cam3.img_points_meters[p2_idx](1), 0.0);

            (h_targetParams + ti*(pp->numOfParams+1))[12] = MAGMA_C_MAKE(views.cam1.img_points_meters[p3_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[13] = MAGMA_C_MAKE(views.cam1.img_points_meters[p3_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[14] = MAGMA_C_MAKE(views.cam2.img_points_meters[p3_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[15] = MAGMA_C_MAKE(views.cam2.img_points_meters[p3_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[16] = MAGMA_C_MAKE(views.cam3.img_points_meters[p3_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[17] = MAGMA_C_MAKE(views.cam3.img_points_meters[p3_idx](1), 0.0);

            (h_targetParams + ti*(pp->numOfParams+1))[18] = MAGMA_C_MAKE(views.cam1.img_tangents_meters[p1_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[19] = MAGMA_C_MAKE(views.cam1.img_tangents_meters[p1_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[20] = MAGMA_C_MAKE(views.cam2.img_tangents_meters[p1_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[21] = MAGMA_C_MAKE(views.cam2.img_tangents_meters[p1_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[22] = MAGMA_C_MAKE(views.cam3.img_tangents_meters[p1_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[23] = MAGMA_C_MAKE(views.cam3.img_tangents_meters[p1_idx](1), 0.0);

            (h_targetParams + ti*(pp->numOfParams+1))[24] = MAGMA_C_MAKE(views.cam1.img_tangents_meters[p2_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[25] = MAGMA_C_MAKE(views.cam1.img_tangents_meters[p2_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[26] = MAGMA_C_MAKE(views.cam2.img_tangents_meters[p2_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[27] = MAGMA_C_MAKE(views.cam2.img_tangents_meters[p2_idx](1), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[28] = MAGMA_C_MAKE(views.cam3.img_tangents_meters[p2_idx](0), 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[29] = MAGMA_C_MAKE(views.cam3.img_tangents_meters[p2_idx](1), 0.0);

            (h_targetParams + ti*(pp->numOfParams+1))[30] = MAGMA_C_MAKE(100.0, 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[31] = MAGMA_C_MAKE(10.0, 0.0);
            (h_targetParams + ti*(pp->numOfParams+1))[32] = MAGMA_C_MAKE(10.0, 0.0);

            (h_targetParams + ti*(pp->numOfParams+1))[33] = MAGMA_C_MAKE(1.0, 0.0);

            //std::cout << "Target parameters #" << ti << std::endl;
            //magma_cprint(pp->numOfParams+1, 1, h_targetParams+ti*(pp->numOfParams+1), (pp->numOfParams+1));

            //> Compute parameter difference
            for (int di = 0; di < pp->numOfParams; di++) (h_params_diff + ti*(pp->numOfParams+1))[di] = (h_targetParams + ti*(pp->numOfParams+1))[di] - (h_startParams)[di];
            (h_params_diff + ti*(pp->numOfParams+1))[pp->numOfParams] = MAGMA_C_ZERO;
        }
    }

    void RANSAC_System::Transfer_Data_From_CPU_to_GPU(  ) 
    {
        //> Transfer data from CPU memory to GPU memory --
        magma_csetmatrix( N+1, batchCount, h_startSols, (N+1), d_startSols, (N+1), my_queue );
        magma_csetmatrix( N+1, batchCount*RANSAC_Number_Of_Iterations, h_Track, (N+1), d_Track, (N+1), my_queue );
        magma_csetmatrix( ldd_params, 1, h_startParams,  ldd_params, d_startParams,  ldd_params, my_queue );
        magma_csetmatrix( ldd_params*RANSAC_Number_Of_Iterations, 1, h_targetParams, ldd_params*RANSAC_Number_Of_Iterations, d_targetParams, ldd_params*RANSAC_Number_Of_Iterations, my_queue );
        magma_csetmatrix( ldd_params*RANSAC_Number_Of_Iterations, 1, h_params_diff,  ldd_params*RANSAC_Number_Of_Iterations, d_diffParams,   ldd_params*RANSAC_Number_Of_Iterations, my_queue );

        magma_csetmatrix( N, N*batchCount, h_cgesvA, N, d_cgesvA, ldda, my_queue );
        magma_csetmatrix( N, batchCount,   h_cgesvB, N, d_cgesvB, lddb, my_queue );
        magma_csetmatrix( batchCount*RANSAC_Number_Of_Iterations, 1, h_path_converge_flag, batchCount*RANSAC_Number_Of_Iterations, d_path_converge_flag, batchCount*RANSAC_Number_Of_Iterations, my_queue );

        magma_isetmatrix( size_Hx, 1, h_Hx_idx, size_Hx, d_Hx_idx, rnded_size_Hx, my_queue );
        magma_isetmatrix( size_Ht, 1, h_Ht_idx, size_Ht, d_Ht_idx, rnded_size_Ht, my_queue );

        //> Connect pointer to 2d arrays
        magma_cset_pointer( d_startSols_array, d_startSols, (N+1), 0, 0, (N+1),  batchCount, my_queue );
        magma_cset_pointer( d_Track_array,     d_Track,     (N+1), 0, 0, (N+1),  batchCount*RANSAC_Number_Of_Iterations, my_queue );
        magma_cset_pointer( d_cgesvA_array,    d_cgesvA,    ldda,  0, 0, ldda*N, batchCount, my_queue );
        magma_cset_pointer( d_cgesvB_array,    d_cgesvB,    lddb,  0, 0, ldda,   batchCount, my_queue );
    }

    void RANSAC_System::Solve_Relative_Pose( magmaHCWrapper::Problem_Params* pp )
    {
        
        gpu_time = magmaHCWrapper::kernel_HC_Solver_trifocal_2op1p_30_direct_param_homotopy
                   (my_queue, ldda, N, pp->numOfParams, batchCount, d_startSols_array, d_Track_array, 
                    d_startParams, d_targetParams, d_cgesvA_array, d_cgesvB_array,
                    d_diffParams, d_Hx_idx, d_Ht_idx, d_path_converge_flag);
    }

    void RANSAC_System::Transfer_Data_From_GPU_to_CPU() 
    {
        //> Check returns from the GPU kernel
        magma_cgetmatrix( (N+1), batchCount*RANSAC_Number_Of_Iterations,   d_Track, (N+1), h_track_sols,   (N+1), my_queue );
        magma_cgetmatrix( batchCount*RANSAC_Number_Of_Iterations, 1, d_path_converge_flag, batchCount*RANSAC_Number_Of_Iterations, h_path_converge_flag,  batchCount*RANSAC_Number_Of_Iterations, my_queue );
        //batchCount*RANSAC_Number_Of_Iterations

        if (DEBUG) {
            magma_cgetmatrix( N,     batchCount,   d_cgesvB, lddb, h_cgesvB_verify, N,    my_queue );
            magma_cgetmatrix( N,     N*batchCount, d_cgesvA, ldda, h_cgesvA_verify, N,    my_queue );
        }
    }

    void RANSAC_System::Transform_Solutions_To_Relative_Poses( TrifocalViewsWrapper::Trifocal_Views views ) 
    {
        MultiviewGeometryUtil::multiview_geometry_util mg_util;
        float norm_t21, norm_t31;
        Eigen::Vector3d T21_;
        Eigen::Vector3d T31_;

        //> Loop over all RANSAC iterations
        for (int ri = 0; ri < RANSAC_Number_Of_Iterations; ri++) {

            //> Loop over all paths
            for (int bs = 0; bs < batchCount; bs++) {
                
                if (MAGMA_C_REAL((h_path_converge_flag + ri*batchCount)[bs]) == 1) {

                    //> Check the imaginary part of the two relative rotations
                    int small_imag_part_counter = 0;
                    for (int vi = 0; vi < 6; vi++) {
                        if ( fabs(MAGMA_C_IMAG((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[vi])) < IMAG_PART_TOL ) small_imag_part_counter++;
                    }

                    //> Pick the solution if all the imaginary parts of the two relative rotations are small enough
                    if ( small_imag_part_counter == 6 ) {

                        //> First normalize the translation part
                        //> T21
                        t21(0) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[18]);
                        t21(1) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[19]);
                        t21(2) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[20]);
                        T21_ = mg_util.Normalize_Translation_Vector( t21 );
                        normalized_t21.push_back( T21_ );
                        
                        //> T31
                        t31(0) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[21]);
                        t31(1) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[22]);
                        t31(2) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[23]);
                        T31_ = mg_util.Normalize_Translation_Vector( t31 );
                        normalized_t31.push_back( T31_ );
                        
                        //> R21
                        r21(0) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[24]);
                        r21(1) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[25]);
                        r21(2) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[26]);
                        R21.push_back( mg_util.Cayley_To_Rotation_Matrix( r21 ) );

                        //> R31
                        r31(0) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[27]);
                        r31(1) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[28]);
                        r31(2) = MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[29]);
                        R31.push_back( mg_util.Cayley_To_Rotation_Matrix( r31 ) );

                        //> Essential Matrices
                        E21.push_back( mg_util.getEssentialMatrix( R21.back(), normalized_t21.back()) );
                        E31.push_back( mg_util.getEssentialMatrix( R31.back(), normalized_t31.back()) );

                        //> Fundamental Matrices
                        F21.push_back( mg_util.getFundamentalMatrix( views.inv_K, R21.back(), normalized_t21.back()) );
                        F31.push_back( mg_util.getFundamentalMatrix( views.inv_K, R31.back(), normalized_t31.back()) );

                        /*if (fabs(fabs(r31(2)) - 0.49545) < 0.01 ) {
                            std::cout << r21(0) << std::endl << r21(1) << std::endl << r21(2) << std::endl;
                            std::cout << r31(0) << std::endl << r31(1) << std::endl << r31(2) << std::endl;
                            std::cout << "(ri, bs, Index) = (" << ri << ", " << bs << ", " << F31.size() << ")" << std::endl;
                            std::cout << std::endl;
                            std::cout << R21[R21.size()-1] << std::endl << std::endl;
                            //std::cout << R21.back() << std::endl;
                            std::cout << T21_ << std::endl << std::endl;
                            std::cout << F21[F21.size()-1] << std::endl << std::endl;
                            std::cout << F31.back() << std::endl << std::endl;
                        }*/
                    }
                }
            }
        }
    }
    
    void RANSAC_System::Find_Solution_With_Maximal_Inliers( TrifocalViewsWrapper::Trifocal_Views views ) 
    {
        Number_Of_Points = views.cam1.img_perturbed_points_pixels.size();
        std::cout << "Number of poses = " << F31.size() << std::endl;

        int Number_Of_Inliers;
        int Maximal_Number_of_Inliers = 0;
        int Pose_Index_with_Maximal_Number_of_Inliers;
        float numerOfDist_21, numerOfDist_31;
        float denomOfDist_21, denomOfDist_31;
        float dist_21, dist_31;

        float epip_coeff_a21, epip_coeff_b21, epip_coeff_c21;
        float epip_coeff_a31, epip_coeff_b31, epip_coeff_c31;
        float epip_coeff_a21_bar, epip_coeff_b21_bar;
        float epip_coeff_a31_bar, epip_coeff_b31_bar;

        //> Loop over all poses
        for (int si = 0; si < F21.size(); si++) {

            Number_Of_Inliers = 0;
            //> Loop over all correspondences
            for (int pi = 0; pi < Number_Of_Points; pi++) {

                //> Let's do it in pixels
                //  Compute epipoloar line equation and coefficients
                Eigen::Vector3d Homogeneous_Point_in_Pixels_View1{views.cam1.img_perturbed_points_pixels[pi](0), views.cam1.img_perturbed_points_pixels[pi](1), 1.0};
                Eigen::Vector3d Epipolar_Line_Coefficients21 = F21[si]*Homogeneous_Point_in_Pixels_View1;
                Eigen::Vector3d Epipolar_Line_Coefficients31 = F31[si]*Homogeneous_Point_in_Pixels_View1;

                epip_coeff_a21 = Epipolar_Line_Coefficients21(0);
                epip_coeff_b21 = Epipolar_Line_Coefficients21(1);
                epip_coeff_c21 = Epipolar_Line_Coefficients21(2);

                epip_coeff_a31 = Epipolar_Line_Coefficients31(0);
                epip_coeff_b31 = Epipolar_Line_Coefficients31(1);
                epip_coeff_c31 = Epipolar_Line_Coefficients31(2);

                //> The corresponding matching points in camera 2
                epip_coeff_a21_bar = epip_coeff_a21 * views.cam2.img_perturbed_points_pixels[pi](0);
                epip_coeff_b21_bar = epip_coeff_b21 * views.cam2.img_perturbed_points_pixels[pi](1);

                //> The corresponding matching points in camera 2
                epip_coeff_a31_bar = epip_coeff_a31 * views.cam3.img_perturbed_points_pixels[pi](0);
                epip_coeff_b31_bar = epip_coeff_b31 * views.cam3.img_perturbed_points_pixels[pi](1);

                /*
                numerOfDist = abs(A_ep + B_it + C);
                denomOfDist = A.^2 + B.^2;
                denomOfDist = sqrt(denomOfDist);
                */
                //> Distance to the epipolar line in view 2 and 3
                numerOfDist_21 = fabs( epip_coeff_a21_bar + epip_coeff_b21_bar + epip_coeff_c21 );
                denomOfDist_21 = sqrt( epip_coeff_a21*epip_coeff_a21 + epip_coeff_b21*epip_coeff_b21 );
                dist_21 = numerOfDist_21 / denomOfDist_21;

                numerOfDist_31 = fabs( epip_coeff_a31_bar + epip_coeff_b31_bar + epip_coeff_c31 );
                denomOfDist_31 = sqrt( epip_coeff_a31*epip_coeff_a31 + epip_coeff_b31*epip_coeff_b31 );
                dist_31 = numerOfDist_31 / denomOfDist_31;

                if ( dist_21 <= SAMPSON_ERROR_THRESH && dist_31 <= SAMPSON_ERROR_THRESH ) {
                    Number_Of_Inliers++;
                }
            }
            if ( Number_Of_Inliers > Maximal_Number_of_Inliers ) {
                Maximal_Number_of_Inliers = Number_Of_Inliers;
                Pose_Index_with_Maximal_Number_of_Inliers = si;
            }
        } 

        //> Assign final RANSAC solution
        final_R21 = R21[ Pose_Index_with_Maximal_Number_of_Inliers ];
        final_R31 = R31[ Pose_Index_with_Maximal_Number_of_Inliers ];
        final_T21 = normalized_t21[ Pose_Index_with_Maximal_Number_of_Inliers ];
        final_T31 = normalized_t31[ Pose_Index_with_Maximal_Number_of_Inliers ];

        std::cout << "> Maximal Number of inliers: " << Maximal_Number_of_Inliers << std::endl;
        std::cout << "> Pose Index: " << Pose_Index_with_Maximal_Number_of_Inliers << std::endl;
        std::cout << "> Pose with Maximal Number of Inliers: " << std::endl;
        std::cout << "> R21: " << std::endl << R21[Pose_Index_with_Maximal_Number_of_Inliers] << std::endl;
        std::cout << "> T21: " << std::endl << normalized_t21[Pose_Index_with_Maximal_Number_of_Inliers] << std::endl;
        std::cout << "> R31: " << std::endl << R31[Pose_Index_with_Maximal_Number_of_Inliers] << std::endl;
        std::cout << "> T31: " << std::endl << normalized_t31[Pose_Index_with_Maximal_Number_of_Inliers] << std::endl;
        std::cout << "> F21: " << std::endl << F21[Pose_Index_with_Maximal_Number_of_Inliers] << std::endl;
    }

    bool RANSAC_System::Solution_Residual_From_GroundTruths( TrifocalViewsWrapper::Trifocal_Views views )
    {
        Eigen::Vector3d euler_ang_GT_21 = views.R21.eulerAngles(0, 1, 2);
        Eigen::Vector3d euler_ang_GT_31 = views.R31.eulerAngles(0, 1, 2);
        Eigen::Vector3d euler_ang_final_21 = final_R21.eulerAngles(0, 1, 2);
        Eigen::Vector3d euler_ang_final_31 = final_R31.eulerAngles(0, 1, 2);
        Rotation_Residual = euler_ang_GT_21 - euler_ang_final_21;
        Translation_Residual = euler_ang_GT_31 - euler_ang_final_31;

        std::cout << "> Euler angles residual: " << std::endl << Rotation_Residual << std::endl;
        std::cout << "> Translation residual: " << std::endl << Translation_Residual << std::endl;

        //> TODO
        return 0;
    }

    void RANSAC_System::Write_Solutions_To_Files( std::ofstream &GPUHC_Solution_File ) 
    {
        int num_of_convergence;
        for (int ri = 0; ri < RANSAC_Number_Of_Iterations; ri++)
        {
            num_of_convergence = 0;
            for (int bs = 0; bs < batchCount; bs++) {
                GPUHC_Solution_File << std::setprecision(10);
                
                //if (MAGMA_C_REAL((h_cgesvB_verify + bs * N)[0]) == 1) {
                if (MAGMA_C_REAL((h_path_converge_flag + ri*batchCount)[bs]) == 1) {

                    //GPUHC_Solution_File << bs << "\t" << MAGMA_C_REAL((h_cgesvB_verify + bs*N)[0]) << "\t" << MAGMA_C_IMAG((h_cgesvB_verify + bs*N)[0]) << "\n";
                    //GPUHC_Solution_File << MAGMA_C_IMAG((h_cgesvB_verify + bs*N)[0]) << "\t" << std::setprecision(20) << (gpu_time)*1000 << "\n";
                    for (int vs = 0; vs < N; vs++) {
                        GPUHC_Solution_File << std::setprecision(20) << MAGMA_C_REAL((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[vs]) << "\t";
                        GPUHC_Solution_File << std::setprecision(20) << MAGMA_C_IMAG((h_track_sols + ri*batchCount*(N+1) + bs * (N+1))[vs]) << "\n";
                    }
                    GPUHC_Solution_File << "\n";
                    num_of_convergence++;
                }
            }
            //std::cout<< "Number of convergence: " << num_of_convergence <<std::endl;
            GPUHC_Solution_File << "==============================================================\n";
        }
        
        //> Show how much time GPU takes for solving the problem
        printf("============== GPU time (ms) ==============\n");
        printf("%7.2f (ms)\n", (gpu_time)*1000);
        printf("===========================================\n\n");
    }

    void RANSAC_System::Free_Memories() 
    {
        magma_free_cpu( h_startSols     );
        magma_free_cpu( h_Track         );
        magma_free_cpu( h_startParams   );
        magma_free_cpu( h_targetParams  );
        magma_free_cpu( h_phc_coeffs_Hx );
        magma_free_cpu( h_phc_coeffs_Ht );
        magma_free_cpu( h_Hx_idx );
        magma_free_cpu( h_Ht_idx );

        magma_queue_destroy( my_queue );

        magma_free_cpu( h_cgesvA );
        magma_free_cpu( h_cgesvB );
        magma_free_cpu( h_cgesvA_verify );
        magma_free_cpu( h_cgesvB_verify );
        magma_free_cpu( h_track_sols );
        magma_free_cpu( h_path_converge_flag );

        magma_free( d_startSols );
        magma_free( d_Track );
        magma_free( d_startParams );
        magma_free( d_targetParams );
        magma_free( d_cgesvA );
        magma_free( d_cgesvB );
        magma_free( d_startSols_array );
        magma_free( d_Track_array );
        magma_free( d_cgesvA_array );
        magma_free( d_cgesvB_array );
        magma_free( d_Hx_idx );
        magma_free( d_Ht_idx );
        magma_free( d_path_converge_flag );

        magma_free_cpu( h_params_diff );
        magma_free( d_diffParams );

        fflush( stdout );
        printf( "\n" );
        magma_finalize();
    }
}

#endif
