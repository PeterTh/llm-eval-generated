cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;75-virtual" CACHE STRING "CUDA architectures")
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Keep explicit floating-point rounding points and optimize host mesh setup.
target_compile_options(roomsim PRIVATE -O3 --fmad=false -Xcompiler=-march=native,-Wall,-Wextra)
