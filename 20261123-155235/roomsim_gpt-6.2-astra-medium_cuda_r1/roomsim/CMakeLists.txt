cmake_minimum_required(VERSION 3.20)
# Prefer the active CUDA toolkit when several versions are installed.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(CMAKE_CUDA_COMPILER nvcc HINTS /usr/local/cuda/bin)
endif()
project(roomsim_benchmark LANGUAGES CUDA)

set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_options(roomsim PRIVATE -O3 -Xcompiler=-march=native)
