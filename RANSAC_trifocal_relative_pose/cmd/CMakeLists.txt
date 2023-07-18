
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#> main lib
find_package(Threads REQUIRED)
link_libraries(ransac_trifocal_relative_pose)

add_executable(ransac_trifocal_relative_pose-main ransac_trifocal_relative_pose_solver.cpp )

set_property(TARGET ransac_trifocal_relative_pose-main
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ransac_trifocal_relative_pose-main Threads::Threads)

#set_target_properties(ransac_trifocal_relative_pose-main PROPERTIES LINKER_LANGUAGE CXX)
#target_link_libraries(ransac_trifocal_relative_pose-main Threads::Threads)