#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
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
//> nvidia cuda
//#include <cuda.h>
//#include <cuda_runtime.h>

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

#include "../ransac_trifocal_relative_pose_solver/Views.h"

//> magma
//#include "magma_v2.h"

//> magma
//#include "magmaHC/magmaHC-problems.cuh"

//> p2c
//#include "magmaHC/const-matrices/p2c-symbolic_3views_4pts.h"

//> Synthetic Curve Dataset
std::string Synthetic_Curve_Dataset_Path = "/oscar/data/bkimia/cchien3/synthcurves-multiview-3d-dataset/spherical-ascii-100_views-perturb-radius_sigma10-normal_sigma0_01rad-minsep_15deg-no_two_cams_colinear_with_object/";

//> global repo directory
std::string repo_dir = "/oscar/data/bkimia/cchien3/Trifocal_Relative_Pose_Estimator/RANSAC_trifocal_relative_pose/";

int main(int argc, char **argv) {
  /*argc; ++argv;
  std::string arg;
  int argIndx = 0;
  int argTotal = 4;
  std::string HC_problem = "default";

  if (argc) {
    arg = std::string(*argv);
    if (arg == "-h" || arg == "--help") {
      magmaHCWrapper::print_usage();
      exit(1);
    }
    else if (argc <= argTotal) {
      while(argIndx <= argTotal-1) {
        if (arg == "-p" || arg == "--problem") {
          argv++;
          arg = std::string(*argv);
          HC_problem = arg;
          argIndx+=2;
          break;
        }
        else {
          std::cerr<<"invalid input arguments! See examples: \n";
          magmaHCWrapper::print_usage();
          exit(1);
        }
        argv++;
      }
    }
    else if (argc > argTotal) {
      std::cerr<<"too many arguments!\n";
      magmaHCWrapper::print_usage();
      exit(1);
    }
  }
  else {
    magmaHCWrapper::print_usage();
    exit(1);
  }*/

  //> Read the curve dataset
  //points2D = load([dataPath 'frame_' num2str(view,'%04.f') '-pts-2D.txt']);
  //tangent = load([dataPath 'frame_' num2str(view,'%04.f') '-tgts-2D.txt']);

  //> View indices
  int View1_Indx = 80;
  int View2_Indx = 82;
  int View3_Indx = 84;

  //> Curve indices
  int Curve1_Indx = 10;
  int Curve2_Indx = 12;
  int Curve3_Indx = 14;

  float outlier_ratio = 0.01;
  int RANSAC_Number_Of_Iterations = 500;

  //> Define constant view variables
  magmaHCWrapper::Trifocal_Views views(View1_Indx, View2_Indx, View3_Indx, Curve1_Indx, Curve2_Indx, Curve3_Indx, Synthetic_Curve_Dataset_Path);
  views.Read_In_Dataset_ImagePoints();
  views.Read_In_Dataset_ImageTangents();
  views.Read_In_Dataset_CameraMatrices();
  views.Add_Noise_to_Points_on_Curves( outlier_ratio );


  //> Start RANSAC Computations
  for (int iter = 0; iter < 1; iter++) {

  }
  
  /*std::cout << "C1:" << std::endl;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << views.cam1.abs_R(i,j) << "\t";
    }
    std::cout << std::endl;
  }*/

  //std::cout << "Image points in view 1:" << std::endl;
  //std::cout << views.cam1.img_points.size() << std::endl;
  
  /*std::cout << "Image points on curve in view 1:" << std::endl;
  std::cout << views.cam1.img_points_on_curve.size() << std::endl;
  for (int i = 0; i < views.cam1.img_points_on_curve.size(); i++) {
    std::cout << "(" << views.cam1.img_points_on_curve[i](0) << ", " << views.cam1.img_points_on_curve[i](1) << ")\n";
  }*/
  
  //> 

  
  /*for (int i = 0; i < 10; i++) {
    std::cout << "(" << views.cam1.img_points[i](0) << ", " << views.cam1.img_points[i](1) << ")" << std::endl;
  }

  std::cout << views.cam1.img_points.size() << std::endl;*/
  



/*
  //> Write successful HC track solutions to files
  std::ofstream track_sols_file;
  std::string write_sols_file_dir = repo_dir;
  write_sols_file_dir.append("GPU_Converged_HC_tracks.txt");
  track_sols_file.open(write_sols_file_dir);
  if ( !track_sols_file.is_open() )
    std::cout<<"files " << write_sols_file_dir << " cannot be opened!"<<std::endl;

  magmaFloatComplex *h_startSols;
  magmaFloatComplex *h_Track;
  magmaFloatComplex *h_startParams;
  magmaFloatComplex *h_targetParams;
  magmaFloatComplex *h_phc_coeffs_Hx;
  magmaFloatComplex *h_phc_coeffs_Ht;
  magma_int_t *h_Hx_idx;
  magma_int_t *h_Ht_idx;

  //> files to be read
  std::string repo_root_dir = repo_dir;
  repo_dir.append("problems/");
  std::string problem_filename = repo_dir.append(HC_problem);

  //> declare class objects (put the long lasting object in dynamic memory)
  magmaHCWrapper::problem_params* pp = new magmaHCWrapper::problem_params;
  //magmaHCWrapper::const_mats* cm = new magmaHCWrapper::const_mats;

  pp->define_problem_params(problem_filename, HC_problem);

  //> allocate tracks and coeffs arrays in cpu
  magma_cmalloc_cpu( &h_startSols,    pp->numOfTracks*(pp->numOfVars+1) );
  magma_cmalloc_cpu( &h_Track,        pp->numOfTracks*(pp->numOfVars+1) );
  magma_cmalloc_cpu( &h_startParams,  pp->numOfParams );
  magma_cmalloc_cpu( &h_targetParams, pp->numOfParams );

  magma_cmalloc_cpu( &h_phc_coeffs_Hx, (pp->numOfCoeffsFromParams+1)*(pp->max_orderOf_t+1) );
  magma_cmalloc_cpu( &h_phc_coeffs_Ht, (pp->numOfCoeffsFromParams+1)*(pp->max_orderOf_t) );
  magma_imalloc_cpu( &h_Hx_idx,        pp->numOfVars*pp->numOfVars*pp->Hx_maximal_terms*pp->Hx_maximal_parts );
  magma_imalloc_cpu( &h_Ht_idx,        pp->numOfVars*pp->Ht_maximal_terms*pp->Ht_maximal_parts );

  // =============================================================================
  //> read files: start solutions, start coefficients, and target parameters
  // =============================================================================
  std::string targetParam_filename_test = problem_filename;
  std::string startParams_filename_test = problem_filename;
  std::string startSols_filename_test = problem_filename;
  startSols_filename_test.append("/start_sols.txt");
  targetParam_filename_test.append("/target_params.txt");
  startParams_filename_test.append("/start_params.txt");
  std::fstream startCoef_file;
  std::fstream targetParams_file;
  std::fstream startSols_file;
  bool read_success = 0;
  bool start_sols_read_success = 0;
  bool start_coeffs_read_success = 0;
  bool targetParams_read_success = 0;
  
  float s_real, s_imag;
  int d = 0, i = 0; 
  startSols_file.open(startSols_filename_test, std::ios_base::in);
  if (!startSols_file) {
    std::cerr << "problem start solutions file not existed!\n";
  }
  else {
    while (startSols_file >> s_real >> s_imag) {
      (h_startSols + i * (pp->numOfVars+1))[d] = MAGMA_C_MAKE(s_real, s_imag);
      (h_Track + i * (pp->numOfVars+1))[d] = MAGMA_C_MAKE(s_real, s_imag);
      if (d < pp->numOfVars-1) {
        d++;
      }
      else {
        d = 0;
        i++;
      }
    }
    for(int k = 0; k < pp->numOfTracks; k++) {
      (h_startSols + k * (pp->numOfVars+1))[pp->numOfVars] = MAGMA_C_MAKE(1.0, 0.0);
      (h_Track + k * (pp->numOfVars+1))[pp->numOfVars] = MAGMA_C_MAKE(1.0, 0.0);
    }
    start_sols_read_success = 1;
  }

  //> read start system coefficients
  d = 0;
  startCoef_file.open(startParams_filename_test, std::ios_base::in);
  if (!startCoef_file) {
    std::cerr << "problem start coefficients file not existed!\n";
  }
  else {
    while (startCoef_file >> s_real >> s_imag) {
      (h_startParams)[d] = MAGMA_C_MAKE(s_real, s_imag);
      d++;
    }
    start_coeffs_read_success = 1;
  }

  d = 0;
  targetParams_file.open(targetParam_filename_test, std::ios_base::in);
  if (!targetParams_file) {
    std::cerr << "problem target parameters file not existed!\n";
  }
  else {
    while (targetParams_file >> s_real >> s_imag) {
      (h_targetParams)[d] = MAGMA_C_MAKE(s_real, s_imag);
      d++;
    }
    targetParams_read_success = 1;
  }

  //>-------------------------------------------------------------------------------------------------
  bool Hx_file_read_success = false;
  bool Ht_file_read_success = false;
  
  std::string filename_Hx = problem_filename;
  std::string filename_Ht = problem_filename;
  filename_Hx.append("/Hx_idx.txt");
  filename_Ht.append("/Ht_idx.txt");
  std::fstream Hx_idx_file;
  std::fstream Ht_idx_file;
  
  //> 4) read Hx index matrix, if required
  int index;
  d = 0;
  Hx_idx_file.open(filename_Hx, std::ios_base::in);
  if (!Hx_idx_file) {
    std::cerr << "problem Hx index matrix file not existed!\n";
  }
  else {
    while (Hx_idx_file >> index) {
      (h_Hx_idx)[d] = index;
      d++;
    }
    Hx_file_read_success = 1;
  }
  //> 5) read Ht index matrix
  d = 0;
  Ht_idx_file.open(filename_Ht, std::ios_base::in);
  if (!Ht_idx_file) {
    std::cerr << "problem Ht index matrix file not existed!\n";
  }
  else {
    while (Ht_idx_file >> index) {
      (h_Ht_idx)[d] = index;
      d++;
    }
    Ht_file_read_success = 1;
  }

  //> numerical params2coeffs
  if (HC_problem == "six_lines_generalized_cam") {
    //magmaHCWrapper::p2c_numerical_six_lines_generalized_cam(h_phc_coeffs_Hx, h_phc_coeffs_Ht);
    magmaHCWrapper::p2c_numerical_six_lines_generalized_cam_v2(h_phc_coeffs_Hx, h_phc_coeffs_Ht);
  }
  else if (HC_problem == "six_lines_16") {
    magmaHCWrapper::p2c_symbolic_six_lines_16(h_startParams, h_targetParams, h_phc_coeffs_Hx, h_phc_coeffs_Ht);
  }
  else if (HC_problem == "3views_4pts") {
    magmaHCWrapper::p2c_symbolic_3views_4pts(h_startParams, h_targetParams, h_phc_coeffs_Hx, h_phc_coeffs_Ht);
  }
  
  read_success = (start_sols_read_success && start_coeffs_read_success && targetParams_read_success && Hx_file_read_success && Ht_file_read_success);

  //> call homotopy continuation solver
  if (read_success) {
    magmaHCWrapper::homotopy_continuation_solver(h_startSols, h_Track, h_startParams, h_targetParams, h_Hx_idx, h_Ht_idx, h_phc_coeffs_Hx, h_phc_coeffs_Ht, pp, HC_problem, track_sols_file);
  }
  else {
    std::cout<<"read files failed!"<<std::endl;
    exit(1);
  }

  delete pp;
  //delete cm;
  magma_free_cpu( h_startSols );
  magma_free_cpu( h_Track );
  magma_free_cpu( h_startParams );
  magma_free_cpu( h_targetParams );
  magma_free_cpu( h_phc_coeffs_Hx );
  magma_free_cpu( h_phc_coeffs_Ht );

  magma_free_cpu( h_Hx_idx );
  magma_free_cpu( h_Ht_idx );

  track_sols_file.close();
  */

  return 0;
}
