cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# General C++ flags
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI and OpenMP
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Try to enable CUDA if available; if found, compile with CUDA and define USE_CUDA
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compiler detected: ${CMAKE_CUDA_COMPILER}")
    set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(spmv PRIVATE USE_CUDA)
    # Make nvcc use OpenMP in host compiler
    target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp -O3>)
endif()
