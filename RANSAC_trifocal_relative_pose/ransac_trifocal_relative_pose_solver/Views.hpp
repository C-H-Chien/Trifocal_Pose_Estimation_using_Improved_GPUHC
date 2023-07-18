#ifndef VIEWS_HPP
#define VIEWS_HPP
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

//#include "magma_v2.h"

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

namespace TrifocalViewsWrapper {
    struct View1 {
        std::vector<Eigen::Vector2d> img_points_pixels;            //> All 2d image points on curves
        std::vector<Eigen::Vector2d> img_tangents_pixels;          //> All 2d image tangents on curves
        std::vector<Eigen::Vector2d> img_perturbed_points_pixels;            //> All 2d perturbed image points on curves
        std::vector<Eigen::Vector2d> img_perturbed_tangents_pixels;          //> All 2d perturbed image tangents on curves
        std::vector<Eigen::Vector2d> img_points_meters;
        std::vector<Eigen::Vector2d> img_tangents_meters;
        std::vector<Eigen::Vector2d> img_perturbed_points_meters;
        std::vector<Eigen::Vector2d> img_perturbed_tangents_meters;
        Eigen::Matrix3d abs_R;
        Eigen::Vector3d abs_C;
        Eigen::Vector3d abs_T;
    };

    struct View2 {
        std::vector<Eigen::Vector2d> img_points_pixels;
        std::vector<Eigen::Vector2d> img_tangents_pixels;
        std::vector<Eigen::Vector2d> img_perturbed_points_pixels;
        std::vector<Eigen::Vector2d> img_perturbed_tangents_pixels;
        std::vector<Eigen::Vector2d> img_points_meters;
        std::vector<Eigen::Vector2d> img_tangents_meters;
        std::vector<Eigen::Vector2d> img_perturbed_points_meters;
        std::vector<Eigen::Vector2d> img_perturbed_tangents_meters;
        Eigen::Matrix3d abs_R;
        Eigen::Vector3d abs_C;
        Eigen::Vector3d abs_T;
    };

    struct View3 {
        std::vector<Eigen::Vector2d> img_points_pixels;
        std::vector<Eigen::Vector2d> img_tangents_pixels;
        std::vector<Eigen::Vector2d> img_perturbed_points_pixels;
        std::vector<Eigen::Vector2d> img_perturbed_tangents_pixels;
        std::vector<Eigen::Vector2d> img_points_meters;
        std::vector<Eigen::Vector2d> img_tangents_meters;
        std::vector<Eigen::Vector2d> img_perturbed_points_meters;
        std::vector<Eigen::Vector2d> img_perturbed_tangents_meters;
        Eigen::Matrix3d abs_R;
        Eigen::Vector3d abs_C;
        Eigen::Vector3d abs_T;
    };

    
    class Trifocal_Views {


    public:
        
        Trifocal_Views(int, int, int, int, int, int, std::string);

        View1 cam1;
        View2 cam2;
        View3 cam3;
        Eigen::Matrix3d K;
        Eigen::Matrix3d inv_K;
        Eigen::Matrix3d R21;
        Eigen::Matrix3d R31;
        Eigen::Vector3d T21;
        Eigen::Vector3d T31;
        Eigen::Matrix3d F21;
        Eigen::Matrix3d F31;
        int Number_Of_Outliers;
        
        void Read_In_Dataset_ImagePoints();
        void Read_In_Dataset_ImageTangents();
        void Read_In_Dataset_CameraMatrices();
        void Add_Noise_to_Points_on_Curves();

        void Compute_Trifocal_Relative_Pose_Ground_Truth();

        int getRandPermNum(std::vector<int>& v);
        void generateRandPermIndices(int n, std::vector<int> & All_Indices);
        void Convert_From_Pixels_to_Meters();

    private:
        int View1_Index;
        int View2_Index;
        int View3_Index;
        int Curve1_Index;
        int Curve2_Index;
        int Curve3_Index;
        std::string Dataset_Path;

        std::string View1_Indx_Str;
        std::string View2_Indx_Str;
        std::string View3_Indx_Str;

        

        std::vector<int> Curve_Index_Collection_List;
    };

}


#endif
