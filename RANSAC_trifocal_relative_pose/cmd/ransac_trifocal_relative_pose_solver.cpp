#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <assert.h>
#include <cassert>
#include <string>
// =======================================================================================================
// main function
//
// Modifications
//    Chiang-Heng Chien  22-11-07    Created a Parameter HC Branch for Minimal Problems Arised from a 
//                                   Generalized Camera Model
//    Chiang-Heng Chien  22-11-23    Add Six Lines With Six Unknowns Problem
//    Chiang-Heng Chien  22-01-17    Add 3-Views With 4-Points Problem
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================
#include "../ransac_trifocal_relative_pose_solver/Views.hpp"

//> magma
#include "magma_v2.h"

//> magma
#include "../ransac_trifocal_relative_pose_solver/Problem_Params.hpp"

//> p2c
#include "../ransac_trifocal_relative_pose_solver/gpu_solver/polys_params2coeffs/p2c_trifocal_2op1p_30.h"

#include "../ransac_trifocal_relative_pose_solver/definitions.h"
#include "../ransac_trifocal_relative_pose_solver/HC_data_reader.hpp"
#include "../ransac_trifocal_relative_pose_solver/RANSAC_System_MB.hpp"
#include "../ransac_trifocal_relative_pose_solver/Files_for_TEST.hpp"

int main(int argc, char **argv) {

  //> View indices
  int View1_Indx = 80;
  int View2_Indx = 82;
  int View3_Indx = 84;

  //> Curve indices (don't care for the moment)
  int Curve1_Indx = 10;
  int Curve2_Indx = 12;
  int Curve3_Indx = 14;

  //> Define constant view variables
  TrifocalViewsWrapper::Trifocal_Views views(View1_Indx, View2_Indx, View3_Indx, Curve1_Indx, Curve2_Indx, Curve3_Indx, CURVE_DATASET_DIR);
  views.Read_In_Dataset_ImagePoints();
  views.Read_In_Dataset_ImageTangents();
  views.Read_In_Dataset_CameraMatrices();
  views.Add_Noise_to_Points_on_Curves( );
  views.Convert_From_Pixels_to_Meters();
  views.Compute_Trifocal_Relative_Pose_Ground_Truth();
  /*std::cout << "> Ground Truths: " << std::endl;
  std::cout << "  R21: " << std::endl;
  std::cout << views.R21 << std::endl;
  std::cout << "  T21: " << std::endl;
  std::cout << views.T21 << std::endl;
  std::cout << "  R31: " << std::endl;
  std::cout << views.R31 << std::endl;
  std::cout << "  T31: " << std::endl;
  std::cout << views.T31 << std::endl;
  std::cout << "  F21: " << std::endl;
  std::cout << views.F21 << std::endl;
  std::cout << "  F31: " << std::endl;
  std::cout << views.F31 << std::endl;*/

  std::cout << views.cam1.img_points_pixels.size() << std::endl;

  //> Assertion check for the deinitions variables
  if( MULTIPLES_OF_BATCHCOUNT * MULTIPLES_OF_TRACKING_PER_WARP == RANSAC_Number_Of_Iterations ) {}
  else { std::cout << "FAILURE: Argument parameters incorrect!" << std::endl; exit(1); }

  //> Files to be read
  std::string repo_root_dir = REPO_DIR;
  std::string HC_problem = "trifocal_2op1p";
  std::string problem_filename = repo_root_dir.append(HC_problem);

  //> declare class objects (put the long lasting object in dynamic memory)
  magmaHCWrapper::Problem_Params* pp = new magmaHCWrapper::Problem_Params;

  //> Get problem parameters
  pp->define_problem_params(problem_filename, HC_problem);

  TEST_WITH_WRITTEN_FILES::write_files_for_test Write_Test_Data_to_Files;

  double gpu_time_accumulation = 0.0;
  double max_gpu_time = 0;
  double min_gpu_time = 10000;
  int find_true_solution_counter = 0;
  //int TOTAL_TEST_TIMES = 100;
  for (int ti = 0; ti < TOTAL_TEST_TIMES; ti++) {
    //> Initialize RANSAC Scheme
    //RANSAC_Estimator::RANSAC_System Solve_by_RANSAC( pp );
    RANSAC_Estimator::RANSAC_System_MB Solve_by_RANSAC( pp );

    //> Allocate necessary arrays
    Solve_by_RANSAC.Array_Memory_Allocations( pp, true );

    // =============================================================================
    //> READ FILES: start solutions, start parameters, and target parameters
    // =============================================================================
    //> For RANSAC purpose, we only need one start system (start system = start solutions + start parameters)
    HC_Data_Import::HC_Data_Reader HC_Data_Reader(problem_filename);

    std::cout << "Reading data from files ..." << std::endl;
    bool is_start_sols_loaded      = HC_Data_Reader.Read_Start_System_Solutions(   Solve_by_RANSAC.h_startSols, Solve_by_RANSAC.h_Track, pp, true );
    bool is_start_params_loaded    = HC_Data_Reader.Read_Start_System_Parameters(  Solve_by_RANSAC.h_startParams,  pp, true );
    bool is_Jacob_eval_indx_loaded = HC_Data_Reader.Read_Hx_Ht_Indices( Solve_by_RANSAC.h_Hx_idx, Solve_by_RANSAC.h_Ht_idx, pp );
    bool read_success = is_start_sols_loaded & is_start_params_loaded & is_Jacob_eval_indx_loaded;

    //if (DEBUG) magma_cprint((pp->numOfVars+1), 1, Solve_by_RANSAC.h_startSols, (pp->numOfVars+1));
    
    Solve_by_RANSAC.Prepare_Target_Params( views, pp );
    //if (DEBUG) magma_cprint(pp->numOfParams+1, 1, Solve_by_RANSAC.h_targetParams+0*(pp->numOfParams+1), (pp->numOfParams+1));
    
    std::cout << "Transfering data from CPU to GPU ..." << std::endl;
    Solve_by_RANSAC.Transfer_Data_From_CPU_to_GPU( );

    Solve_by_RANSAC.Solve_Relative_Pose( pp );
    gpu_time_accumulation += Solve_by_RANSAC.gpu_time;
    if ( Solve_by_RANSAC.gpu_time > max_gpu_time ) max_gpu_time = Solve_by_RANSAC.gpu_time;
    if ( Solve_by_RANSAC.gpu_time < min_gpu_time ) min_gpu_time = Solve_by_RANSAC.gpu_time;

    Solve_by_RANSAC.Transfer_Data_From_GPU_to_CPU();

    #if TEST_BLOCK_CYCLE_TIME
    Solve_by_RANSAC.Push_Block_Cycle_Time();
    #else
    Solve_by_RANSAC.Transform_Solutions_To_Relative_Poses( views );
    Solve_by_RANSAC.Find_Solution_With_Maximal_Inliers( views );
    bool is_True_Solution_Found = Solve_by_RANSAC.Solution_Residual_From_GroundTruths( views );
    if (is_True_Solution_Found) find_true_solution_counter++;

    Write_Test_Data_to_Files.write_Pose_Residuals( 
         Solve_by_RANSAC.Stacked_R21_Residuals, Solve_by_RANSAC.Stacked_R31_Residuals,
         Solve_by_RANSAC.Stacked_T21_Residuals, Solve_by_RANSAC.Stacked_T31_Residuals );

    #endif

    #if WRITE_SOLUTION_TO_FILE
    //> Write converged HC track solutions to files
    std::ofstream GPUHC_Solution_File;
    std::string write_sols_file_dir = REPO_DIR;
    write_sols_file_dir.append("GPU_Converged_HC_tracks.txt");
    GPUHC_Solution_File.open(write_sols_file_dir);
    if ( !GPUHC_Solution_File.is_open() ) std::cout << "files " << write_sols_file_dir << " cannot be opened!" << std::endl;
    Solve_by_RANSAC.Write_Solutions_To_Files( GPUHC_Solution_File ) ;
    GPUHC_Solution_File.close();
    #endif

    #if TEST_COLLECT_HC_STEPS
    Write_Test_Data_to_Files.write_true_solution_HC_steps( Solve_by_RANSAC.true_solution_hc_steps );
    #endif 

    #if TEST_ALL_POSITIVE_DEPTHS_AT_END_ZONE
    Write_Test_Data_to_Files.write_time_when_depths_are_positive( Solve_by_RANSAC.t_when_depths_are_positive_for_actual_sol );
    #endif

    #if TEST_BLOCK_CYCLE_TIME
    Write_Test_Data_to_Files.write_block_cycle_times( Solve_by_RANSAC.all_Block_Cycle_Times );
    #endif

    Solve_by_RANSAC.Free_Memories();
  }

  #if TEST_BLOCK_CYCLE_TIME
  std::cout << "Finish timing the block cycles!" << std::endl;
  #else
  //> Show multiple times results
  if (TOTAL_TEST_TIMES > 0) {
    std::cout << "======================== INFORMATION ==========================" << std::endl;
    std::cout << "> RANSAC Iterations:              " << RANSAC_Number_Of_Iterations << std::endl;
    std::cout << "> Noise std of Point Position:    " << NOISE_DISTRIBUTION_STD_INLIERS_POINTS << std::endl;
    std::cout << "> Noise std of Point Orientation: " << NOISE_DISTRIBUTION_STD_INLIERS_TANGENTS << std::endl;
    std::cout << "> Outlier Ratio (%):              " << OUTLIER_RATIO*100 << std::endl;
    std::cout << "========================== RESULTS ============================" << std::endl;
    std::cout << "> Average Success Rate (%):       " << ((double)find_true_solution_counter / (double)TOTAL_TEST_TIMES)*100 << std::endl;
    std::cout << "> Average Time Cost (ms):         " << ((double)gpu_time_accumulation / (double)TOTAL_TEST_TIMES)*1000 << std::endl;
    std::cout << "> Max/Min Time Cost (ms):         " << max_gpu_time*1000 << " / " << min_gpu_time*1000 << std::endl;
  }
  #endif

  //> Write Ground Truths to a File
  Write_Test_Data_to_Files.write_GroundTruths(views.R21, views.R31, views.T21, views.T31);
    
  Write_Test_Data_to_Files.close_all_files();

  delete pp;

  return 0;  
}
