cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

# Keep the existing source and executable names; compile the source with nvcc.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)
# Avoid fast-math and reassociation: Monte Carlo samples and correlation ties
# must retain the reference implementation's numerical behavior.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)
target_link_libraries(roomsim PRIVATE m)
