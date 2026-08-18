cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

find_package(OpenMP REQUIRED)
find_package(MPI QUIET COMPONENTS CXX)
target_link_libraries(spmv PRIVATE OpenMP::OpenMP_CXX)
if(MPI_CXX_FOUND)
    target_link_libraries(spmv PRIVATE MPI::MPI_CXX)
    target_compile_definitions(spmv PRIVATE SPMV_USE_MPI=1)
endif()

# Compile the existing source as CUDA when a toolkit is available.  This keeps
# the executable name and source layout unchanged on CUDA clusters.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    target_compile_definitions(spmv PRIVATE SPMV_USE_CUDA=1)
endif()

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(spmv PRIVATE m)
