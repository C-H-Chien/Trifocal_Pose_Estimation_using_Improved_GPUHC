#ifndef VIEWS_CPP
#define VIEWS_CPP
// =======================================================================
//
// Modifications
//    Chiang-Heng Chien  23-06-15:   Intiailly Created
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================
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

#include "Views.hpp"
#include "definitions.h"
#include "util.hpp"

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

namespace TrifocalViewsWrapper {
    
    //extern "C"
    Trifocal_Views::Trifocal_Views(
        int cam1_idx, int cam2_idx, int cam3_idx, 
        int curve1_idx, int curve2_idx, int curve3_idx, 
        std::string dataset_dir)
    :View1_Index(cam1_idx), View2_Index(cam2_idx), View3_Index(cam3_idx), 
     Curve1_Index(curve1_idx), Curve2_Index(curve2_idx), Curve3_Index(curve3_idx),
     Dataset_Path(dataset_dir) 
     {
        //> Calibration matrix
        /*K(0,0) = 2584.932509819502; //> fx
        K(0,1) = 0;
        K(0,2) = 249.771375872214;  //> cx
        K(1,0) = 0;
        K(1,1) = 2584.791860605769; //> fy
        K(1,2) = 278.312679379194;  //> cy
        K(2,0) = 0;
        K(2,1) = 0;
        K(2,2) = 0;*/

        int n_zero = 4;
        View1_Indx_Str = std::string(n_zero - std::min(n_zero, (int)(std::to_string(View1_Index).length())), '0') + std::to_string(View1_Index);
        View2_Indx_Str = std::string(n_zero - std::min(n_zero, (int)(std::to_string(View2_Index).length())), '0') + std::to_string(View2_Index);
        View3_Indx_Str = std::string(n_zero - std::min(n_zero, (int)(std::to_string(View3_Index).length())), '0') + std::to_string(View3_Index);

        //> Read curve indices list
        int readin_curve_idx;
        std::string curveIdx_Path = Dataset_Path + "crv-ids.txt";
        std::fstream Curve_Indices_File;
        Curve_Indices_File.open(curveIdx_Path, std::ios_base::in);
        if (!Curve_Indices_File) {std::cerr << "Path " << curveIdx_Path << " does not exist. \n"; exit(1);}
        else {
            while ( Curve_Indices_File >> readin_curve_idx ) {
                Curve_Index_Collection_List.push_back(readin_curve_idx);
            }
        }
    }

    //extern "C"
    void Trifocal_Views::Read_In_Dataset_ImagePoints() {
        //> Set path to read image 2D points
        std::string View1_Points2D_Path = Dataset_Path + "frame_" + View1_Indx_Str + "-pts-2D.txt";
        std::string View2_Points2D_Path = Dataset_Path + "frame_" + View2_Indx_Str + "-pts-2D.txt";
        std::string View3_Points2D_Path = Dataset_Path + "frame_" + View3_Indx_Str + "-pts-2D.txt";

        //> VIEW 1 Image Points
        double subpix_x, subpix_y;
        //int readin_counter = 0;
        std::fstream View1_Points2D_File;
        View1_Points2D_File.open(View1_Points2D_Path, std::ios_base::in);
        if (!View1_Points2D_File) {std::cerr << "Path " << View1_Points2D_Path << " does not exist. \n"; exit(1);}
        else {
            while ( View1_Points2D_File >> subpix_x >> subpix_y ) {
                Eigen::Vector2d subpix_Curve_Point2D{subpix_x, subpix_y};
                cam1.img_points_pixels.push_back(subpix_Curve_Point2D);

                //> Push the image curve points (three curves) on view 1
                //if ( Curve_Index_Collection_List[readin_counter] == Curve1_Index ) cam1.img_points_pixels_on_curve[0].push_back(subpix_Curve_Point2D);
                //if ( Curve_Index_Collection_List[readin_counter] == Curve2_Index ) cam1.img_points_pixels_on_curve[1].push_back(subpix_Curve_Point2D);
                //if ( Curve_Index_Collection_List[readin_counter] == Curve3_Index ) cam1.img_points_pixels_on_curve[2].push_back(subpix_Curve_Point2D);
                //readin_counter++;
            }
        }

        //> VIEW 2 Image Points
        //readin_counter = 0;
        std::fstream View2_Points2D_File;
        View2_Points2D_File.open(View2_Points2D_Path, std::ios_base::in);
        if (!View2_Points2D_File) {std::cerr << "Path " << View2_Points2D_Path << " does not exist. \n"; exit(1);}
        else {
            while ( View2_Points2D_File >> subpix_x >> subpix_y ) {
                Eigen::Vector2d subpix_Curve_Point2D{subpix_x, subpix_y};
                cam2.img_points_pixels.push_back(subpix_Curve_Point2D);

                //> Push the image curve points (three curves) on view 2
                //if ( Curve_Index_Collection_List[readin_counter] == Curve1_Index ) cam2.img_points_pixels_on_curve[0].push_back(subpix_Curve_Point2D);
                //if ( Curve_Index_Collection_List[readin_counter] == Curve2_Index ) cam2.img_points_pixels_on_curve[1].push_back(subpix_Curve_Point2D);
                //if ( Curve_Index_Collection_List[readin_counter] == Curve3_Index ) cam2.img_points_pixels_on_curve[2].push_back(subpix_Curve_Point2D);
                //readin_counter++;
            }
        }

        //> VIEW 3 Image Points
        //readin_counter = 0;
        std::fstream View3_Points2D_File;
        View3_Points2D_File.open(View3_Points2D_Path, std::ios_base::in);
        if (!View3_Points2D_File) {std::cerr << "Path " << View3_Points2D_Path << " does not exist. \n"; exit(1);}
        else {
            while ( View3_Points2D_File >> subpix_x >> subpix_y ) {
                Eigen::Vector2d subpix_Curve_Point2D{subpix_x, subpix_y};
                cam3.img_points_pixels.push_back(subpix_Curve_Point2D);

                //> Push the image curve points (three curves) on view 3
                //if ( Curve_Index_Collection_List[readin_counter] == Curve1_Index ) cam3.img_points_pixels_on_curve[0].push_back(subpix_Curve_Point2D);
                //if ( Curve_Index_Collection_List[readin_counter] == Curve2_Index ) cam3.img_points_pixels_on_curve[1].push_back(subpix_Curve_Point2D);
                //if ( Curve_Index_Collection_List[readin_counter] == Curve3_Index ) cam3.img_points_pixels_on_curve[2].push_back(subpix_Curve_Point2D);
                //readin_counter++;
            }
        }
    }

    //extern "C"
    void Trifocal_Views::Read_In_Dataset_ImageTangents() {
        //> Set path to read image 2D tangents
        std::string View1_Tangents2D_Path = Dataset_Path + "frame_" + View1_Indx_Str + "-tgts-2D.txt";
        std::string View2_Tangents2D_Path = Dataset_Path + "frame_" + View2_Indx_Str + "-tgts-2D.txt";
        std::string View3_Tangents2D_Path = Dataset_Path + "frame_" + View3_Indx_Str + "-tgts-2D.txt";

        //> VIEW 1 Image Tangent
        //int readin_counter = 0;
        double subpix_x, subpix_y;
        std::fstream View1_Tangents2D_File;
        View1_Tangents2D_File.open(View1_Tangents2D_Path, std::ios_base::in);
        if (!View1_Tangents2D_File) {std::cerr << "Path " << View1_Tangents2D_Path << " does not exist. \n"; exit(1);}
        else {
            while ( View1_Tangents2D_File >> subpix_x >> subpix_y ) {
                Eigen::Vector2d subpix_Curve_Tangent2D{subpix_x, subpix_y};
                cam1.img_tangents_pixels.push_back(subpix_Curve_Tangent2D);

                //> Push the image curve tangents on view 1
                //if ( Curve_Index_Collection_List[readin_counter] == Curve1_Index ) cam1.img_tangents_pixels_on_curve.push_back(subpix_Curve_Tangent2D);
                //readin_counter++;
            }
        }

        //> VIEW 2 Image Tangent
        //readin_counter = 0;
        std::fstream View2_Tangents2D_File;
        View2_Tangents2D_File.open(View2_Tangents2D_Path, std::ios_base::in);
        if (!View2_Tangents2D_File) {std::cerr << "Path " << View2_Tangents2D_Path << " does not exist. \n"; exit(1);}
        else {
            while ( View2_Tangents2D_File >> subpix_x >> subpix_y ) {
                Eigen::Vector2d subpix_Curve_Tangent2D{subpix_x, subpix_y};
                cam2.img_tangents_pixels.push_back(subpix_Curve_Tangent2D);

                //> Push the image curve tangents on view 2
                //if ( Curve_Index_Collection_List[readin_counter] == Curve2_Index ) cam2.img_tangents_pixels_on_curve.push_back(subpix_Curve_Tangent2D);
                //readin_counter++;
            }
        }

        //> VIEW 3 Image Tangent
        //readin_counter = 0;
        std::fstream View3_Tangents2D_File;
        View3_Tangents2D_File.open(View3_Tangents2D_Path, std::ios_base::in);
        if (!View3_Tangents2D_File) {std::cerr << "Path " << View3_Tangents2D_Path << " does not exist. \n"; exit(1);}
        else {
            while ( View3_Tangents2D_File >> subpix_x >> subpix_y ) {
                Eigen::Vector2d subpix_Curve_Tangent2D{subpix_x, subpix_y};
                cam3.img_tangents_pixels.push_back(subpix_Curve_Tangent2D);

                //> Push the image curve tangents on view 3
                //if ( Curve_Index_Collection_List[readin_counter] == Curve3_Index ) cam3.img_tangents_pixels_on_curve.push_back(subpix_Curve_Tangent2D);
                //readin_counter++;
            }
        }
    }

    void Trifocal_Views::Read_In_Dataset_CameraMatrices() {
        //> Set path to read camera poses and calibration matrix
        std::string View1_ExtrinsicMatrix_Path = Dataset_Path + "frame_" + View1_Indx_Str + ".extrinsic";
        std::string View2_ExtrinsicMatrix_Path = Dataset_Path + "frame_" + View2_Indx_Str + ".extrinsic";
        std::string View3_ExtrinsicMatrix_Path = Dataset_Path + "frame_" + View3_Indx_Str + ".extrinsic";
        std::string Calibration_Matrix_Path    = Dataset_Path + "calib.intrinsic";
        
        //> VIEW 1 Rotation and Translation
        std::fstream View1_Pose_File;
        View1_Pose_File.open(View1_ExtrinsicMatrix_Path, std::ios_base::in);
        if (!View1_Pose_File) {std::cerr << "Path " << View1_ExtrinsicMatrix_Path << " does not exist. \n"; exit(1);}
        else {
            View1_Pose_File >> cam1.abs_R(0,0) >> cam1.abs_R(0,1) >> cam1.abs_R(0,2);
            View1_Pose_File >> cam1.abs_R(1,0) >> cam1.abs_R(1,1) >> cam1.abs_R(1,2);
            View1_Pose_File >> cam1.abs_R(2,0) >> cam1.abs_R(2,1) >> cam1.abs_R(2,2);
            View1_Pose_File >> cam1.abs_C(0) >> cam1.abs_C(1) >> cam1.abs_C(2);
        }
        //> Compute the translation vector for view 1
        cam1.abs_T = -cam1.abs_R * cam1.abs_C;

        //> VIEW 2 Rotation and Translation
        std::fstream View2_Pose_File;
        View2_Pose_File.open(View2_ExtrinsicMatrix_Path, std::ios_base::in);
        if (!View2_Pose_File) {std::cerr << "Path " << View2_ExtrinsicMatrix_Path << " does not exist. \n"; exit(1);}
        else {
            View2_Pose_File >> cam2.abs_R(0,0) >> cam2.abs_R(0,1) >> cam2.abs_R(0,2);
            View2_Pose_File >> cam2.abs_R(1,0) >> cam2.abs_R(1,1) >> cam2.abs_R(1,2);
            View2_Pose_File >> cam2.abs_R(2,0) >> cam2.abs_R(2,1) >> cam2.abs_R(2,2);
            View2_Pose_File >> cam2.abs_C(0) >> cam2.abs_C(1) >> cam2.abs_C(2);
        }
        //> Compute the translation vector for view 2
        cam2.abs_T = -cam2.abs_R * cam2.abs_C;

        //> VIEW 3 Rotation and Translation
        std::fstream View3_Pose_File;
        View3_Pose_File.open(View3_ExtrinsicMatrix_Path, std::ios_base::in);
        if (!View3_Pose_File) {std::cerr << "Path " << View3_ExtrinsicMatrix_Path << " does not exist. \n"; exit(1);}
        else {
            View3_Pose_File >> cam3.abs_R(0,0) >> cam3.abs_R(0,1) >> cam3.abs_R(0,2);
            View3_Pose_File >> cam3.abs_R(1,0) >> cam3.abs_R(1,1) >> cam3.abs_R(1,2);
            View3_Pose_File >> cam3.abs_R(2,0) >> cam3.abs_R(2,1) >> cam3.abs_R(2,2);
            View3_Pose_File >> cam3.abs_C(0) >> cam3.abs_C(1) >> cam3.abs_C(2);
        }
        //> Compute the translation vector for view 3
        cam3.abs_T = -cam3.abs_R * cam3.abs_C;

        //> Calibration Matrix
        std::fstream Camera_Intrinsic_Matrix_File;
        Camera_Intrinsic_Matrix_File.open(Calibration_Matrix_Path, std::ios_base::in);
        if (!Camera_Intrinsic_Matrix_File) {std::cerr << "Path " << Calibration_Matrix_Path << " does not exist. \n"; exit(1);}
        else {
            Camera_Intrinsic_Matrix_File >> K(0,0) >> K(0,1) >> K(0,2);
            Camera_Intrinsic_Matrix_File >> K(1,0) >> K(1,1) >> K(1,2);
            Camera_Intrinsic_Matrix_File >> K(2,0) >> K(2,1) >> K(2,2);
        }

        //> Get the inverse of the calibration matrix
        inv_K = K.inverse();

        //if (DEBUG) std::cout << "Calibration matrix: " << K << std::endl;
    }

    //extern "C"
    void Trifocal_Views::Add_Noise_to_Points_on_Curves() {
        //> Random number generator credit: https://en.cppreference.com/w/cpp/numeric/random
        //> Seed with a real random value, if available
        std::random_device r;

        //> Set Perturbations
        //std::default_random_engine generator_pts(1);
        //std::default_random_engine generator_ore(1);
        //std::default_random_engine generator_pts( r() );
        //std::default_random_engine generator_ore( r() );

        //> For point positions
        std::normal_distribution<double> Noise_Distribution_On_Point_Inliers(0, NOISE_DISTRIBUTION_STD_INLIERS_POINTS);
        std::normal_distribution<double> Noise_Distribution_On_Point_Outliers(0, NOISE_DISTRIBUTION_STD_OUTLIERS_POINTS);

        //> For tangent orientations
        std::normal_distribution<double> Noise_Distribution_On_Tangent_Inliers(0, NOISE_DISTRIBUTION_STD_INLIERS_TANGENTS);
        std::normal_distribution<double> Noise_Distribution_On_Tangent_Outliers(0, NOISE_DISTRIBUTION_STD_OUTLIERS_TANGENTS);
        std::uniform_int_distribution<int> theta(-10000, 10000);

        //> Create sizes for the vector structures
        const int Number_Of_Points = cam1.img_points_pixels.size();
        cam1.img_perturbed_points_pixels.resize(Number_Of_Points);
        cam2.img_perturbed_points_pixels.resize(Number_Of_Points);
        cam3.img_perturbed_points_pixels.resize(Number_Of_Points);
        cam1.img_perturbed_tangents_pixels.resize(Number_Of_Points);
        cam2.img_perturbed_tangents_pixels.resize(Number_Of_Points);
        cam3.img_perturbed_tangents_pixels.resize(Number_Of_Points);

        //> Generate Outlier Point Indices;
        std::vector<int> All_Points_Indices;
        generateRandPermIndices( Number_Of_Points, All_Points_Indices );
        Number_Of_Outliers = (int)(cam1.img_points_pixels.size() * OUTLIER_RATIO);
        std::cout << "Number of Outliers = " << Number_Of_Outliers << std::endl << "Outliers:" << std::endl;

        //> List of Indices Indicating whether the point index is an outlier or not
        //> Note: Variable-sized Array Must Be Initialized Properly 
        bool Boolean_Outlier_Indices[Number_Of_Points];
        std::fill_n (Boolean_Outlier_Indices, Number_Of_Points, false);     //> Make all false

        for (int oi = 0; oi < Number_Of_Outliers; oi++) { 
            int indx = All_Points_Indices[oi];
            Boolean_Outlier_Indices[indx] = true;
            //std::cout << indx << std::endl;
        }

        //> Perturb All Points on Curve 1 Across All Views
        //> TODO: Make sure that the perturbed points are withing view boundaries
        double Inlier_Mag, Outlier_Mag, angle;
        double Inlier_Tangent_Ore, Outlier_Tangent_Ore;
        int outlier_counter = 0;
        bool is_outlier;
        for (int pi = 0; pi < Number_Of_Points; pi++) {

            //> Check whether the current index is an outlier or not
            is_outlier = Boolean_Outlier_Indices[pi];
            
            //> Perturb both inlier and outliers for each view individually so that the pertubation are not the same across all views
            //> 1) View 1
            if (is_outlier) {
                std::default_random_engine generator_pts( r() );
                std::default_random_engine generator_ore( r() );

                //> Point position perturbation
                Outlier_Mag = Noise_Distribution_On_Point_Outliers(generator_pts);
                angle =  M_PI * (theta(generator_pts) / 10000.0);
                cam1.img_perturbed_points_pixels[pi]   = cam1.img_points_pixels[pi] + Eigen::Vector2d(Outlier_Mag*cos(angle), Outlier_Mag*sin(angle));
                //cam1.img_perturbed_tangents_pixels[pi] = cam1.img_tangents_pixels[pi];

                //> Point tangent perturbation
                double orientation_deg = atan ( cam1.img_tangents_pixels[pi](1) / cam1.img_tangents_pixels[pi](0) ) * 180.0 / M_PI;
                orientation_deg += Noise_Distribution_On_Tangent_Outliers(generator_ore);
                double orientation_rad = orientation_deg * M_PI / 180.0;
                cam1.img_perturbed_tangents_pixels[pi] = Eigen::Vector2d{ cos(orientation_rad), sin(orientation_rad) };

                /*if (outlier_counter < 10) {
                    std::cout << "O: (" << cam1.img_points_pixels[pi](0) << ", " << cam1.img_points_pixels[pi](1) << ") -> (";
                    std::cout << cam1.img_perturbed_points_pixels[pi](0) << ", " << cam1.img_perturbed_points_pixels[pi](1) << ")" << std::endl;
                }*/
            }
            else {
                std::default_random_engine generator_pts( r() );
                std::default_random_engine generator_ore( r() );

                //> Point position perturbation
                Inlier_Mag         = Noise_Distribution_On_Point_Inliers(generator_pts);
                angle =  M_PI * (theta(generator_pts) / 10000.0);
                cam1.img_perturbed_points_pixels[pi]   = cam1.img_points_pixels[pi] + Eigen::Vector2d(Inlier_Mag*cos(angle), Inlier_Mag*sin(angle));
                
                //> Point tangent perturbation
                double orientation_deg = atan ( cam1.img_tangents_pixels[pi](1) / cam1.img_tangents_pixels[pi](0) ) * 180.0 / M_PI;
                orientation_deg += Noise_Distribution_On_Tangent_Inliers(generator_ore);
                double orientation_rad = orientation_deg * M_PI / 180.0;
                cam1.img_perturbed_tangents_pixels[pi] = Eigen::Vector2d{ cos(orientation_rad), sin(orientation_rad) };
                //cam1.img_perturbed_tangents_pixels[pi] = cam1.img_tangents_pixels[pi];

                if (pi < 10) {
                    std::cout << "I: (" << cam1.img_points_pixels[pi](0) << ", " << cam1.img_points_pixels[pi](1) << ") -> (";
                    std::cout << cam1.img_perturbed_points_pixels[pi](0) << ", " << cam1.img_perturbed_points_pixels[pi](1) << ")" << std::endl;
                }
            }

            //> 2) View 2
            if (is_outlier) {
                std::default_random_engine generator_pts( r() );
                std::default_random_engine generator_ore( r() );

                //> Point position perturbation
                Outlier_Mag = Noise_Distribution_On_Point_Outliers(generator_pts);
                angle =  M_PI * (theta(generator_pts) / 10000.0);
                cam2.img_perturbed_points_pixels[pi]   = cam2.img_points_pixels[pi] + Eigen::Vector2d(Outlier_Mag*cos(angle), Outlier_Mag*sin(angle));
                
                //> Point tangent perturbation
                //cam2.img_perturbed_tangents_pixels[pi] = cam2.img_tangents_pixels[pi];
                double orientation_deg = atan ( cam2.img_tangents_pixels[pi](1) / cam2.img_tangents_pixels[pi](0) ) * 180.0 / M_PI;
                orientation_deg += Noise_Distribution_On_Tangent_Outliers(generator_ore);
                double orientation_rad = orientation_deg * M_PI / 180.0;
                cam2.img_perturbed_tangents_pixels[pi] = Eigen::Vector2d{ cos(orientation_rad), sin(orientation_rad) };
            }
            else {
                std::default_random_engine generator_pts( r() );
                std::default_random_engine generator_ore( r() );

                //> Point position perturbation
                Inlier_Mag  = Noise_Distribution_On_Point_Inliers(generator_pts);
                angle =  M_PI * (theta(generator_pts) / 10000.0);
                cam2.img_perturbed_points_pixels[pi]   = cam2.img_points_pixels[pi] + Eigen::Vector2d(Inlier_Mag*cos(angle), Inlier_Mag*sin(angle));
                
                //> Point tangent perturbation
                //cam2.img_perturbed_tangents_pixels[pi] = cam2.img_tangents_pixels[pi];
                double orientation_deg = atan ( cam2.img_tangents_pixels[pi](1) / cam2.img_tangents_pixels[pi](0) ) * 180.0 / M_PI;
                orientation_deg += Noise_Distribution_On_Tangent_Inliers(generator_ore);
                double orientation_rad = orientation_deg * M_PI / 180.0;
                cam2.img_perturbed_tangents_pixels[pi] = Eigen::Vector2d{ cos(orientation_rad), sin(orientation_rad) };
            }

            //> 2) View 3
            if (is_outlier) {
                std::default_random_engine generator_pts( r() );
                std::default_random_engine generator_ore( r() );

                //> Point position perturbation
                Outlier_Mag = Noise_Distribution_On_Point_Outliers(generator_pts);
                angle =  M_PI * (theta(generator_pts) / 10000.0);
                cam3.img_perturbed_points_pixels[pi]   = cam3.img_points_pixels[pi] + Eigen::Vector2d(Outlier_Mag*cos(angle), Outlier_Mag*sin(angle));
                
                //> Point tangent perturbation
                double orientation_deg = atan ( cam3.img_tangents_pixels[pi](1) / cam3.img_tangents_pixels[pi](0) ) * 180.0 / M_PI;
                orientation_deg += Noise_Distribution_On_Tangent_Outliers(generator_ore);
                double orientation_rad = orientation_deg * M_PI / 180.0;
                cam3.img_perturbed_tangents_pixels[pi] = Eigen::Vector2d{ cos(orientation_rad), sin(orientation_rad) };
                //cam3.img_perturbed_tangents_pixels[pi] = cam3.img_tangents_pixels[pi];
                
                outlier_counter++;
            }
            else {
                std::default_random_engine generator_pts( r() );
                std::default_random_engine generator_ore( r() );

                //> Point position perturbation
                Inlier_Mag  = Noise_Distribution_On_Point_Inliers(generator_pts);
                angle =  M_PI * (theta(generator_pts) / 10000.0);
                cam3.img_perturbed_points_pixels[pi]   = cam3.img_points_pixels[pi] + Eigen::Vector2d(Inlier_Mag*cos(angle), Inlier_Mag*sin(angle));
                cam3.img_perturbed_tangents_pixels[pi] = cam3.img_tangents_pixels[pi];

                //> Point tangent perturbation
                //cam2.img_perturbed_tangents_pixels[pi] = cam2.img_tangents_pixels[pi];
                double orientation_deg = atan ( cam3.img_tangents_pixels[pi](1) / cam3.img_tangents_pixels[pi](0) ) * 180.0 / M_PI;
                orientation_deg += Noise_Distribution_On_Tangent_Inliers(generator_ore);
                double orientation_rad = orientation_deg * M_PI / 180.0;
                cam3.img_perturbed_tangents_pixels[pi] = Eigen::Vector2d{ cos(orientation_rad), sin(orientation_rad) };
            }
        }
    }

    void Trifocal_Views::Convert_From_Pixels_to_Meters() {
        const int Number_Of_Points = cam1.img_points_pixels.size();
        Eigen::Vector3d Homogeneous_Img_Points_Pixels{0.0, 0.0, 1.0};
        Eigen::Vector3d Homogeneous_Img_Tangents_Pixels{0.0, 0.0, 0.0};
        Eigen::Vector3d Homogeneous_Point_in_Meters, Homogeneous_Tangent_in_Meters;
        Eigen::Vector2d Point_in_Meters, Tangent_in_Meters;
        for (int pi = 0; pi < Number_Of_Points; pi++) {
            //> Unperturbed Points
            //> Camera 1
            Homogeneous_Img_Points_Pixels(0) = cam1.img_points_pixels[pi](0);
            Homogeneous_Img_Points_Pixels(1) = cam1.img_points_pixels[pi](1);
            //Homogeneous_Img_Points_Pixels(2) = 1.0;
            Homogeneous_Point_in_Meters = inv_K * Homogeneous_Img_Points_Pixels;
            cam1.img_points_meters.push_back( {Homogeneous_Point_in_Meters(0), Homogeneous_Point_in_Meters(1) } );

            //> Camera 2
            Homogeneous_Img_Points_Pixels(0) = cam2.img_points_pixels[pi](0);
            Homogeneous_Img_Points_Pixels(1) = cam2.img_points_pixels[pi](1);
            Homogeneous_Point_in_Meters = inv_K * Homogeneous_Img_Points_Pixels;
            cam2.img_points_meters.push_back( { Homogeneous_Point_in_Meters(0), Homogeneous_Point_in_Meters(1) } );

            //> Camera 3
            Homogeneous_Img_Points_Pixels(0) = cam3.img_points_pixels[pi](0);
            Homogeneous_Img_Points_Pixels(1) = cam3.img_points_pixels[pi](1);
            Homogeneous_Point_in_Meters = inv_K * Homogeneous_Img_Points_Pixels;
            cam3.img_points_meters.push_back( { Homogeneous_Point_in_Meters(0), Homogeneous_Point_in_Meters(1) } );

            //> Perturbed Points
            //> Camera 1
            Homogeneous_Img_Points_Pixels(0) = cam1.img_perturbed_points_pixels[pi](0);
            Homogeneous_Img_Points_Pixels(1) = cam1.img_perturbed_points_pixels[pi](1);
            Homogeneous_Point_in_Meters = inv_K * Homogeneous_Img_Points_Pixels;
            cam1.img_perturbed_tangents_meters.push_back( { Homogeneous_Point_in_Meters(0), Homogeneous_Point_in_Meters(1) } );

            //> Camera 2
            Homogeneous_Img_Points_Pixels(0) = cam2.img_perturbed_points_pixels[pi](0);
            Homogeneous_Img_Points_Pixels(1) = cam2.img_perturbed_points_pixels[pi](1);
            Homogeneous_Point_in_Meters = inv_K * Homogeneous_Img_Points_Pixels;
            cam2.img_perturbed_tangents_meters.push_back( { Homogeneous_Point_in_Meters(0), Homogeneous_Point_in_Meters(1) } );

            //> Camera 3
            Homogeneous_Img_Points_Pixels(0) = cam3.img_perturbed_points_pixels[pi](0);
            Homogeneous_Img_Points_Pixels(1) = cam3.img_perturbed_points_pixels[pi](1);
            Homogeneous_Point_in_Meters = inv_K * Homogeneous_Img_Points_Pixels;
            cam3.img_perturbed_tangents_meters.push_back( { Homogeneous_Point_in_Meters(0), Homogeneous_Point_in_Meters(1) } );

            //> Unperturbed tangents
            //> Camera 1
            Homogeneous_Img_Tangents_Pixels(0) = cam1.img_tangents_pixels[pi](0);
            Homogeneous_Img_Tangents_Pixels(1) = cam1.img_tangents_pixels[pi](1);
            //Homogeneous_Img_Points_Pixels(2) = 1.0;
            Homogeneous_Tangent_in_Meters = inv_K * Homogeneous_Img_Tangents_Pixels;
            cam1.img_tangents_meters.push_back( { Homogeneous_Tangent_in_Meters(0), Homogeneous_Tangent_in_Meters(1) } );
            cam1.img_perturbed_tangents_meters.push_back( { Homogeneous_Tangent_in_Meters(0), Homogeneous_Tangent_in_Meters(1) } );

            //> Camera 2
            Homogeneous_Img_Tangents_Pixels(0) = cam2.img_tangents_pixels[pi](0);
            Homogeneous_Img_Tangents_Pixels(1) = cam2.img_tangents_pixels[pi](1);
            Homogeneous_Tangent_in_Meters = inv_K * Homogeneous_Img_Tangents_Pixels;
            cam2.img_tangents_meters.push_back( { Homogeneous_Tangent_in_Meters(0), Homogeneous_Tangent_in_Meters(1) } );
            cam2.img_perturbed_tangents_meters.push_back( { Homogeneous_Tangent_in_Meters(0), Homogeneous_Tangent_in_Meters(1) } );

            //> Camera 3
            Homogeneous_Img_Tangents_Pixels(0) = cam3.img_tangents_pixels[pi](0);
            Homogeneous_Img_Tangents_Pixels(1) = cam3.img_tangents_pixels[pi](1);
            Homogeneous_Tangent_in_Meters = inv_K * Homogeneous_Img_Tangents_Pixels;
            cam3.img_tangents_meters.push_back( { Homogeneous_Tangent_in_Meters(0), Homogeneous_Tangent_in_Meters(1) } );
            cam3.img_perturbed_tangents_meters.push_back( { Homogeneous_Tangent_in_Meters(0), Homogeneous_Tangent_in_Meters(1)} );
        }
    }

    void Trifocal_Views::Compute_Trifocal_Relative_Pose_Ground_Truth() {
        //R12 = abs_R2 * abs_R1';
        //T12 = -abs_R2 * abs_R1' * abs_T1 + abs_T2;
        MultiviewGeometryUtil::multiview_geometry_util mg_util;

        R21 = cam2.abs_R * cam1.abs_R.transpose();
        R21 = mg_util.Normalize_Rotation_Matrix( R21 );
        T21 = -cam2.abs_R * cam1.abs_R.transpose() * cam1.abs_T + cam2.abs_T;
        T21 = mg_util.Normalize_Translation_Vector( T21 );

        R31 = cam3.abs_R * cam1.abs_R.transpose();
        R31 = mg_util.Normalize_Rotation_Matrix( R31 );
        T31 = -cam3.abs_R * cam1.abs_R.transpose() * cam1.abs_T + cam3.abs_T;
        T31 = mg_util.Normalize_Translation_Vector( T31 );

        F21 = mg_util.getFundamentalMatrix( inv_K, R21, T21 );
        F31 = mg_util.getFundamentalMatrix( inv_K, R31, T31 );
    }

    //extern "C"
    int Trifocal_Views::getRandPermNum(std::vector<int>& v) {
        //> Credit: https://www.geeksforgeeks.org/generate-a-random-permutation-of-1-to-n/
        //> Size of the vector
        int n = v.size();

        //> Generate a random number
        //std::srand(time(NULL));
        std::srand(1);

        //> Make sure the number is within the index range
        int index = rand() % n;

        //> Get random number from the vector
        int num = v[index];

        //> Remove the number from the vector
        std::swap(v[index], v[n - 1]);
        v.pop_back();

        //> Return the removed number
        return num;
    }

    //extern "C"
    void Trifocal_Views::generateRandPermIndices(int n, std::vector<int> & All_Indices) {
        //> Edited from: https://www.geeksforgeeks.org/generate-a-random-permutation-of-1-to-n/
        std::vector<int> v(n);

        // Fill the vector with the values: 1, 2, 3, ..., n
        for (int i = 0; i < n; i++) v[i] = i;

        // get a random number from the vector and print it
        while (v.size()) All_Indices.push_back(getRandPermNum(v));
    }
}

#endif
