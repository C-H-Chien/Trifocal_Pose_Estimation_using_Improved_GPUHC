#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
set ( control_sources
  Views.hpp
  Views.cpp

  #>  input info
  define_params.cpp
  HC_data_reader.cpp
  HC_data_reader.hpp

  #> Macros
  definitions.h

  #> Multiview Geometry Utilities
  util.hpp
  util.cpp

  #> RANSAC Scheme
  RANSAC_System.cpp
  RANSAC_System.hpp

  #>  main gpu code
  Problem_Params.hpp
  HC_Track_General_Problem.cpp
  HC_Track_Chicago_Problem.cpp

  #>  magma kernels
  gpu_solver/gpu-kernels/magmaHC-kernels.h
  #gpu_solver/gpu-kernels/kernel_HC_Solver_trifocal_2op1p.cu
  gpu_solver/gpu-kernels/kernel_HC_Solver_trifocal_2op1p_direct_param_homotopy.cu

  #>  device function
  gpu_solver/gpu-dev-functions/dev-cgesv-batched-small.cuh
  gpu_solver/gpu-dev-functions/dev-get-new-data.cuh
  gpu_solver/gpu-dev-functions/dev-eval-indxing-trifocal_2op1p_30.cuh
  gpu_solver/gpu-dev-functions/dev-eval-indxing-trifocal_2op1p_30_direct_param_homotopy.cuh

  #>  p2c constant matrix
  gpu_solver/polys_params2coeffs/p2c_trifocal_2op1p_30.h

)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/include" )
include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/testing" )
include_directories( "/users/cchien3/data/cchien3/magma/control" )
include_directories( "/users/cchien3/data/cchien3/magma/magmablas" )

add_library( ransac_trifocal_relative_pose ${control_sources} )
target_link_libraries(ransac_trifocal_relative_pose 
        -L/gpfs/runtime/opt/magma-gpu/2.4.0/lib magma
        -L${CUDAPATH}/lib cublas cudart cusparse
        -L/gpfs/runtime/opt/openblas/0.3.7/lib openblas
        -L/usr/lib64 pthread
)
set_target_properties(ransac_trifocal_relative_pose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
