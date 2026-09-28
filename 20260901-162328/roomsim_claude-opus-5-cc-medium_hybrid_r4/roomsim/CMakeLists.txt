cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
  endif()
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# The single translation unit contains the CUDA kernels and is compiled by nvcc.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings.
# Floating point contraction is disabled on the host as well as on the device so that a
# triangle pair yields bit-identical results no matter whether it was evaluated by a CPU
# thread or by a CUDA thread.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3
        --fmad=false
        --extended-lambda
        --expt-relaxed-constexpr
        -Xcompiler=-O3,-march=native,-ffp-contract=off,-Wall,-Wextra,-fopenmp>)

target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
