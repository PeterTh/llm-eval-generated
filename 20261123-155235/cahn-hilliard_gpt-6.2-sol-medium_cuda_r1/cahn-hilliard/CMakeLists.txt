cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(detect_native_cuda_architecture ON)
endif()

project(cahn_hilliard_benchmark LANGUAGES CUDA)

if(detect_native_cuda_architecture AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to target" FORCE)
endif()

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(cahn_hilliard PRIVATE -O3 --fmad=false -Xcompiler=-Wall,-Wextra)
