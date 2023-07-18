#ifndef UTIL_CPP
#define UTIL_CPP
// ====================================================================================================
//
// Modifications
//    Chiang-Heng Chien  23-07-14:   Intiailly Created. Some functions are shifted from my ICCV code.
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

#include "util.hpp"
#include "definitions.h"

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

namespace MultiviewGeometryUtil {
    
    multiview_geometry_util::multiview_geometry_util( ) { }

    Eigen::Matrix3d multiview_geometry_util::Cayley_To_Rotation_Matrix( Eigen::Vector3d r )
    {
        /*
        M = [1+x*x-(y*y+z*z)  2*(x*y-z)         2*(x*z+y);
         2*(x*y+z)        1+y^2-(x*x+z*z)   2*(y*z-x);
         2*(x*z-y)        2*(y*z+x)         1+z*z-(x*x+y*y)];
        */
        Eigen::Matrix3d R;
        R(0,0) = 1 + r(0)*r(0) - (r(1)*r(1) + r(2)*r(2));
        R(0,1) = 2*(r(0)*r(1) - r(2));
        R(0,2) = 2*(r(0)*r(2) + r(1));
        R(1,0) = 2*(r(0)*r(1) + r(2));
        R(1,1) = 1 + r(1)*r(1) - (r(0)*r(0) + r(2)*r(2));
        R(1,2) = 2*(r(1)*r(2) - r(0));
        R(2,0) = 2*(r(0)*r(2) - r(1));
        R(2,1) = 2*(r(1)*r(2) + r(0));
        R(2,2) = 1 + r(2)*r(2) - (r(0)*r(0) + r(1)*r(1));

        //> The rotation matrix R now is up to some scale. Thus we need to normalize it.
        R = Normalize_Rotation_Matrix( R );

        return R;
    }

    Eigen::Matrix3d multiview_geometry_util::Normalize_Rotation_Matrix( Eigen::Matrix3d R ) {
        //> Compute the column vector norms
        float norm_col1 = sqrt(R(0,0)*R(0,0) + R(1,0)*R(1,0) + R(2,0)*R(2,0));
        float norm_col2 = sqrt(R(0,1)*R(0,1) + R(1,1)*R(1,1) + R(2,1)*R(2,1));
        float norm_col3 = sqrt(R(0,2)*R(0,2) + R(1,2)*R(1,2) + R(2,2)*R(2,2));
        R(0,0) /= norm_col1;
        R(1,0) /= norm_col1;
        R(2,0) /= norm_col1;
        R(0,1) /= norm_col2;
        R(1,1) /= norm_col2;
        R(2,1) /= norm_col2;
        R(0,2) /= norm_col3;
        R(1,2) /= norm_col3;
        R(2,2) /= norm_col3;

        //> Check whether R is normalized such that det(R)=1
        assert( fabs(R.determinant() - 1.0) <= IS_SO3_DET_R_TOL );

        return R;
    }

    Eigen::Vector3d multiview_geometry_util::Normalize_Translation_Vector( Eigen::Vector3d T ) {
        double norm_T = T.norm();
        T(0) /= norm_T;
        T(1) /= norm_T;
        T(2) /= norm_T;
        return T;
    }

    Eigen::Matrix3d multiview_geometry_util::getSkewSymmetric(Eigen::Vector3d T) {
        Eigen::Matrix3d T_x = (Eigen::Matrix3d() << 0.,  -T(2),   T(1), T(2),  0.,  -T(0), -T(1),  T(0),   0.).finished();
        return T_x;
    }

    Eigen::Matrix3d multiview_geometry_util::getEssentialMatrix( Eigen::Matrix3d R21, Eigen::Vector3d T21 ) {
        //> E21 = (skew_T(T21)*R21);
        Eigen::Matrix3d T21_x = getSkewSymmetric(T21);
        return T21_x * R21;
    }

    Eigen::Matrix3d multiview_geometry_util::getFundamentalMatrix(Eigen::Matrix3d inverse_K, Eigen::Matrix3d R21, Eigen::Vector3d T21) {
        //> F21 = inv_K'*(skew_T(T21)*R21)*inv_K;
        Eigen::Matrix3d T21_x = getSkewSymmetric(T21);
        return inverse_K.transpose() * (T21_x * R21) * inverse_K;
    }
}

#endif
