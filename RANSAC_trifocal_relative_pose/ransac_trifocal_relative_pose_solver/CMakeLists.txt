
set (control_sources
  Views.h
  Views.cpp
  
  #>  input info
  #define_params_and_read_files.cu

  #>  main gpu code
  #magmaHC-problems.cuh
  #homotopy_continuation_solver.cu

  #>  magma kernels
  #gpu-kernels/magmaHC-kernels.h
  #gpu-kernels/kernel_HC_Solver_six_lines_generalized_cam.cu
  #gpu-kernels/kernel_HC_Solver_six_lines_16.cu
  #gpu-kernels/kernel_HC_Solver_3views_4pts.cu

  #>  device function
  #dev-cgesv-batched-small.cuh
  #dev-get-new-data.cuh
  #gpu-idx-evals/dev-eval-indxing-six_lines_generalized_cam.cuh
  #gpu-idx-evals/dev-eval-indxing-six_lines_16.cuh
  #gpu-idx-evals/dev-eval-indxing-3views_4pts.cuh

  #>  p2c constant matrix
  #const-matrices/p2c-numerical_six_lines_generalized_cam.h
  #const-matrices/p2c-numerical_six_lines_generalized_cam_v2.h
  #const-matrices/p2c-symbolic_six_lines_16.h
  #const-matrices/p2c-symbolic_3views_4pts.h

  #> CPU-HC Solver
  #cpu-solver/cpu-hc-solver/cpu-hc-solver-six_lines_16.cpp
  #cpu-solver/cpu-eval-HxHt/cpu-eval-HxHt-six_lines_16.h
  #cpu-solver/cpu-eval-HxH/cpu-eval-HxH-six_lines_16.h
  #cpu-solver/cpu-solvers.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/include" )
#include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/testing" )
#include_directories( "/users/cchien3/data/cchien3/magma/control" )
#include_directories( "/users/cchien3/data/cchien3/magma/magmablas" )

add_library( ransac_trifocal_relative_pose ${control_sources} )
target_link_libraries(ransac_trifocal_relative_pose 
	PUBLIC OpenMP::OpenMP_CXX
        #-L/gpfs/runtime/opt/magma-gpu/2.4.0/lib magma
        #-L${CUDAPATH}/lib cublas cudart cusparse
        #-L/gpfs/runtime/opt/openblas/0.3.7/lib openblas
        #-L/usr/lib64 pthread
)
set_target_properties(ransac_trifocal_relative_pose PROPERTIES LINKER_LANGUAGE CXX)
#set_target_properties(ransac_trifocal_relative_pose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

