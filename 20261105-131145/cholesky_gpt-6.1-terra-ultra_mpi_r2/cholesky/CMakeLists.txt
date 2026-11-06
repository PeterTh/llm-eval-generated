cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fopenmp-simd" HAS_OPENMP_SIMD)

add_executable(cholesky cholesky.cpp)
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)
if(HAS_OPENMP_SIMD)
    target_compile_options(cholesky PRIVATE -fopenmp-simd)
endif()
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)
