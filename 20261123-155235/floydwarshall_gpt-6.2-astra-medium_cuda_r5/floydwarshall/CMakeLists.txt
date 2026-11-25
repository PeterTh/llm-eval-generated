cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(detect_gpu_architecture ON)
endif()
project(floydwarshall_benchmark LANGUAGES CUDA)

# Prefer native code on the build GPU and retain PTX for newer devices.
if(detect_gpu_architecture AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
    string(REPLACE "-real" "" detected_architectures "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    set(CMAKE_CUDA_ARCHITECTURES "${detected_architectures}" CACHE STRING "CUDA architectures" FORCE)
endif()

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
set_target_properties(floydwarshall PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(floydwarshall PRIVATE -O3)
