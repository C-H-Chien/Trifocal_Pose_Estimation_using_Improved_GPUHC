
set (control_sources

  #> GPU-HC 
  GPU_HC_Solver.hpp
  GPU_HC_Solver.cpp

  #> CPU-HC
  CPU_HC_Solver.hpp
  CPU_HC_Solver.cpp
  cpuhc-solvers/CPUHC_Generic_Solver.cpp
  cpuhc-solvers/CPUHC_Generic_Solver_Eval_by_Indx.cpp

  #> Read problem data
  Data_Reader.hpp
  Data_Reader.cpp

  #> Evaluations on HC solutions
  Evaluations.hpp
  Evaluations.cpp

  #>  magma kernels
  gpu-kernels/magmaHC-kernels.hpp
  gpu-kernels/kernel_GPUHC_trifocal_2op1p_30x30_PH_CodeOpt_TrunPaths.cu
  gpu-kernels/kernel_GPUHC_trifocal_2op1p_30x30_PH_CodeOpt_TrunPaths_Volta.cu
  gpu-kernels/kernel_GPUHC_trifocal_2op1p_30x30_PH_CodeOpt_TrunPaths_TrunRANSAC.cu
  gpu-kernels/kernel_GPUHC_trifocal_2op1p_30x30_PH_CodeOpt_TrunPaths_TrunRANSAC_Volta.cu

  #>  device function
  dev-cgesv-batched-small.cuh
  dev-cgesv-batched-small-sm.cuh
  dev-get-new-data.cuh
  dev-trifocal_2op1p-eval.cuh
  gpu-idx-evals/dev-eval-indxing-trifocal_2op1p_30x30_inline_LimUnroll.cuh
  gpu-idx-evals/dev-eval-indxing-trifocal_2op1p_30x30_LimUnroll_L2Cache.cuh

  #> Jacobian evaluations for CPU-HC
  cpu-jacobian-evals/cpu-eval-dHdX_dHdt_trifocal_2op1p_30x30.hpp
  cpu-jacobian-evals/cpu-eval-dHdX_H_trifocal_2op1p_30x30.hpp
  cpu-jacobian-evals/cpu-eval-indx_trifocal_2op1p_30x30.hpp

  #> Utilities
  util.hpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/include" )
include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/control" )
include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/magmablas" )
include_directories( "/gpfs/data/bkimia/cchien3/yaml-cpp/bin/include" )

add_library( magmaHC ${control_sources} )
target_compile_options(magmaHC PUBLIC -Wall ${OpenMP_CXX_FLAGS})

target_include_directories( magmaHC 
    PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(magmaHC
  PUBLIC OpenMP::OpenMP_CXX
	-L/gpfs/data/bkimia/magma-cchien/lib magma
  -L/gpfs/runtime/opt/openblas/0.3.23-u6k5fey/lib openblas
  -L/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64 yaml-cpp
)
#set_target_properties(magmaHC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



