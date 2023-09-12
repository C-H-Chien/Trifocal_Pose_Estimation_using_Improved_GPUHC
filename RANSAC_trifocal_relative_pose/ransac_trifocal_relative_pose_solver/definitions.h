#ifndef DEFINITION_H
#define DEFINITION_H

//> Change the directory if necessary
//> 1) Synthetic Curve Dataset
#define CURVE_DATASET_DIR          std::string("/oscar/data/bkimia/cchien3/synthcurves-multiview-3d-dataset/spherical-ascii-100_views-perturb-radius_sigma10-normal_sigma0_01rad-minsep_15deg-no_two_cams_colinear_with_object/")
//> 2) Repository Directory
#define REPO_DIR                   std::string("/oscar/data/bkimia/cchien3/Trifocal_Relative_Pose_Estimator/")

//> RANSAC
#define OUTLIER_RATIO               (0.3)     //> Must be within [0, 1]
#define RANSAC_Number_Of_Iterations (20)
#define SAMPSON_ERROR_THRESH        (2)

//> Noise applied to synthetic data for both points and tangents
#define NOISE_DISTRIBUTION_STD_INLIERS_POINTS   (0.0)
#define NOISE_DISTRIBUTION_STD_OUTLIERS_POINTS  (150)
#define NOISE_DISTRIBUTION_STD_INLIERS_TANGENTS   (0.0)
#define NOISE_DISTRIBUTION_STD_OUTLIERS_TANGENTS  (90.0)

//> Returned Solution parameters
#define WRITE_SOLUTION_TO_FILE       (0)
#define IMAG_PART_TOL                (1e-5)             //> Imaginary part tolerance when picking real solutions
#define RESIDUAL_TO_GT_TOL           (1e-4)             //> Residual tolerance of the solution to the ground truth

//> For RANSAC Multiple Batches (MB)
//> MULTIPLES_OF_BATCHCOUNT * MULTIPLES_OF_TRACKING_PER_WARP = RANSAC_Number_Of_Iterations
#define MULTIPLES_OF_BATCHCOUNT        (20)
#define MULTIPLES_OF_TRACKING_PER_WARP (1)

//> Some TESTINGS
#define TOTAL_TEST_TIMES            (100)   //> TEST repeated times, simulating number of RANSAC iterations
#define TEST_COLLECT_HC_STEPS       (1)

//> Some Assertion Checks
#define IS_SO3_DET_R_TOL            (1e-5)              //> Check whether a rotation matrix belongs to SO(3) group, i.e., det(R)=1

#define OUTPUT_WRITE_FOLDER        std::string("outputs_write_files/")

//#define ENABLE_CPUHC_SOLVER        (1)

#define SHOW_LAPACK_NUM_OF_THREADS (0)

#define DEBUG                      (0)

#endif